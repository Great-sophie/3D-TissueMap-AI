"""Training and inference with explicit input channels and sample-level test isolation."""
from pathlib import Path
import json
import random
import numpy as np
import torch
from .data import read_manifest, load_sample, channel_image, normalize_image
from .models import MiniUNet, dice_bce_loss


def _make_tensors(path, nuclear, dim, slice_idx=None):
    image, mask, channels = load_sample(path)
    x = normalize_image(channel_image(image, channels, nuclear))
    y = (mask > 0).astype(np.float32)
    if dim == 2:
        idx = x.shape[0] // 2 if slice_idx is None else slice_idx
        x = x[idx]; y = y[idx]
    # Shapes: [B,C,Z,Y,X] or [B,C,Y,X].
    return torch.from_numpy(x.copy())[None, None], torch.from_numpy(y.copy())[None, None]


def _random_patch(x, y, patch_shape):
    """Apply identical random crops in spatial coordinates to input and mask."""
    size=x.shape[2:]
    if patch_shape is None:
        return x, y
    patch_shape=tuple(int(v) for v in patch_shape[-len(size):])
    if any(p<=0 or p%2 for p in patch_shape):
        raise ValueError("patch sizes must be positive and even")
    bounds=[]
    for n,p in zip(size,patch_shape):
        effective=min(n,p)
        if effective % 2:
            effective-=1
        if effective < 2:
            raise ValueError("Volume too small for pooling")
        start=int(np.random.randint(0,n-effective+1))
        bounds.append(slice(start,start+effective))
    slices=(slice(None),slice(None),*bounds)
    return x[slices],y[slices]


def train_model(manifest, model_dim, output, channel="Nuclear", epochs=2, lr=1e-3, base=8, seed=7, device="auto", patch_shape=(12,48,48), steps_per_sample=2):
    df = read_manifest(manifest)
    train = df[df.split == "train"]
    if len(train) < 1:
        raise ValueError("At least one training sample needed")
    if epochs < 1 or steps_per_sample < 1:
        raise ValueError("epochs and steps_per_sample must both be >= 1")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model = MiniUNet(dim=model_dim, base=base).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train(); losses = []
        for row in train.itertuples():
            img, _, _ = load_sample(row.path)
            for _ in range(steps_per_sample):
                k = int(np.random.randint(img.shape[1])) if model_dim == 2 else None
                x, y = _make_tensors(row.path, channel, model_dim, k)
                x, y = _random_patch(x,y,patch_shape)
                if any(s % 2 for s in x.shape[2:]):
                    raise ValueError("Use even Z/Y/X sizes for MiniUNet; pad/crop before training")
                x, y = x.to(device), y.to(device)
                optim.zero_grad(set_to_none=True)
                logits = model(x)
                loss = dice_bce_loss(logits, y)
                loss.backward(); optim.step()
                losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(losses))})
    checkpoint = output / "model.pt"
    torch.save({"state_dict": model.cpu().state_dict(), "model_dim": model_dim,
                "base": base, "channel": channel, "seed": seed,
                "train_sample_ids": train.sample_id.tolist(), "patch_shape":list(patch_shape),
                "steps_per_sample":steps_per_sample}, checkpoint)
    (output/"history.json").write_text(json.dumps(history, indent=2))
    return checkpoint, history


def _predict_single(model, x, dim, device, patch_shape=None):
    """Predict one canonical ZYX volume. Tiles avoid whole-volume GPU inference."""
    if dim == 2:
        pred = np.zeros_like(x, dtype=np.float32)
        with torch.no_grad():
            for z in range(len(x)):
                patch = torch.from_numpy(x[z].copy())[None, None].to(device)
                pred[z] = torch.sigmoid(model(patch))[0,0].cpu().numpy()
        return pred
    if patch_shape is None:
        patch_shape = (min(16, x.shape[0]), min(64, x.shape[1]), min(64, x.shape[2]))
    patch_shape = tuple(int(v) for v in patch_shape)
    if any(p <= 0 or p % 2 for p in patch_shape):
        raise ValueError("3D patch dimensions must be positive and even")
    total = np.zeros_like(x, dtype=np.float32); counts = np.zeros_like(x, dtype=np.float32)
    def indices(n, p):
        if n <= p:
            return [0]
        stride = max(1, p//2)
        return sorted(set(list(range(0, n-p+1, stride)) + [n-p]))
    with torch.no_grad():
        for z in indices(x.shape[0], patch_shape[0]):
            for y in indices(x.shape[1], patch_shape[1]):
                for xx in indices(x.shape[2], patch_shape[2]):
                    z1,y1,x1 = min(z+patch_shape[0],x.shape[0]),min(y+patch_shape[1],x.shape[1]),min(xx+patch_shape[2],x.shape[2])
                    crop = x[z:z1,y:y1,xx:x1]
                    pad = [(0, max(0,p-s)) for p,s in zip(patch_shape,crop.shape)]
                    crop = np.pad(crop, pad, mode="constant")
                    pred = torch.sigmoid(model(torch.from_numpy(crop.copy())[None,None].to(device)))[0,0].cpu().numpy()
                    pred = pred[:z1-z,:y1-y,:x1-xx]
                    total[z:z1,y:y1,xx:x1] += pred
                    counts[z:z1,y:y1,xx:x1] += 1
    if not np.all(counts > 0):
        raise AssertionError("Sliding window coverage error")
    return total / counts


def predict_manifest(manifest, checkpoint, output, split="test", threshold=.5, patch=(12, 48, 48), device="auto"):
    df = read_manifest(manifest)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniUNet(dim=int(ck["model_dim"]), base=int(ck["base"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    selected = df[df.split == split] if split != "all" else df
    if len(selected) == 0:
        raise ValueError(f"No samples in '{split}'")
    for row in selected.itertuples():
        img, _, names = load_sample(row.path)
        x = normalize_image(channel_image(img, names, ck["channel"]))
        if any(s % 2 for s in x.shape[1:]):
            raise ValueError("Even XY dimensions required. Crop/pad with a documented policy.")
        p = _predict_single(model, x, int(ck["model_dim"]), device, patch)
        np.savez_compressed(output/f"{row.sample_id}.npz", probability=p.astype(np.float32),
                            semantic_mask=(p >= threshold).astype(np.uint8))
    return output
