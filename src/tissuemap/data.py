"""Data ingestion with explicit spatial axes, channel mapping and patient-level split QC."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile

COLUMNS = ["sample_id", "patient_id", "split", "path", "spacing_z_um", "spacing_y_um", "spacing_x_um"]
VALID_SPLITS = {"train", "val", "test"}


def read_manifest(path):
    path = Path(path).resolve()
    df = pd.read_csv(path)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing fields: {sorted(missing)}")
    if df.sample_id.isna().any() or df.sample_id.duplicated().any():
        raise ValueError("sample_id must be present and unique")
    if not set(df.split).issubset(VALID_SPLITS):
        raise ValueError("Allowed split values: train, val, test")
    for name, group in df.groupby("patient_id"):
        if group.split.nunique() > 1:
            raise ValueError(f"Patient leakage: {name} occurs in multiple splits")
    for col in ["spacing_z_um", "spacing_y_um", "spacing_x_um"]:
        if not np.isfinite(df[col]).all() or (df[col] <= 0).any():
            raise ValueError(f"Invalid positive spacing: {col}")
    df["path"] = df.path.map(lambda p: str((path.parent / p).resolve()) if not Path(p).is_absolute() else str(Path(p)))
    missing_files = [p for p in df.path if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing sample files: {missing_files[:3]}")
    return df


def load_sample(path):
    with np.load(path, allow_pickle=False) as f:
        image = f["image"].astype(np.float32)
        mask = f["mask"].astype(np.int32)
        channels = [str(x) for x in f["channels"].tolist()]
    if image.ndim != 4 or mask.ndim != 3 or image.shape[1:] != mask.shape:
        raise ValueError("Expected image (C,Z,Y,X) and instance mask (Z,Y,X), same dimensions")
    if len(channels) != image.shape[0] or len(set(channels)) != len(channels):
        raise ValueError("Channel names must be unique and match image's C dimension")
    if (mask < 0).any() or not np.isfinite(image).all():
        raise ValueError("Masks must be nonnegative integers; image pixels must be finite")
    return image, mask, channels


def channel_image(image, channels, name):
    if name not in channels:
        raise ValueError(f"Channel '{name}' absent. Available: {channels}")
    return image[channels.index(name)]


def normalize_image(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99.5])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def read_stack(path, axes):
    """Read TIFF/NPY and canonicalize axes; do not silently guess channel order."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        arr = tifffile.imread(str(path))
    axes = axes.upper()
    if sorted(axes) != sorted("CZYX") and sorted(axes) != sorted("ZYX"):
        raise ValueError("axes must be a permutation of CZYX or ZYX")
    if arr.ndim != len(axes):
        raise ValueError(f"axes={axes} length does not match tensor rank={arr.ndim}")
    if "C" not in axes:
        arr = np.transpose(arr, [axes.index(a) for a in "ZYX"])[None, ...]
    else:
        arr = np.transpose(arr, [axes.index(a) for a in "CZYX"])
    return arr


def convert_sample(image_file, mask_file, image_axes, mask_axes, channels, sample_id, patient_id, split, spacing, output, manifest):
    if split not in VALID_SPLITS:
        raise ValueError("Invalid split")
    image = read_stack(image_file, image_axes)
    mask = read_stack(mask_file, mask_axes)
    if mask.shape[0] != 1:
        raise ValueError("Mask TIFF must contain exactly 1 channel")
    if image.shape[1:] != mask.shape[1:]:
        raise ValueError("Image/mask spatial dimensions differ: verify aligned 3D registration")
    if len(channels) != image.shape[0]:
        raise ValueError("Explicit --channels count does not match image")
    if not all(s > 0 and np.isfinite(s) for s in spacing):
        raise ValueError("Spacing must be 3 positive micrometer values in Z,Y,X order")
    if not np.issubdtype(mask.dtype, np.integer) or (mask < 0).any():
        raise ValueError("Mask requires nonnegative integer instance IDs")
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, image=image.astype(np.float32), mask=mask[0].astype(np.int32), channels=np.asarray(channels))
    manifest = Path(manifest).resolve(); manifest.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(sample_id=sample_id, patient_id=patient_id, split=split,
                 path=str(output.relative_to(manifest.parent)) if output.is_relative_to(manifest.parent) else str(output),
                 spacing_z_um=spacing[0], spacing_y_um=spacing[1], spacing_x_um=spacing[2])
    if manifest.exists():
        old = pd.read_csv(manifest)
        if sample_id in old.sample_id.values:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        df = pd.concat([old, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    df.to_csv(manifest, index=False)
    read_manifest(manifest)
    return output


def generate_synthetic(out, samples=6, seed=7, shape=(12, 48, 48)):
    """Deterministic artificial data for pipeline smoke tests. Never report as biology."""
    from scipy.ndimage import gaussian_filter
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    rows = []
    for i in range(samples):
        labels = np.zeros(shape, dtype=np.int32)
        img = np.clip(rng.normal(0.045, 0.02, (4, *shape)), 0, None).astype(np.float32)
        for obj in range(1, 11):
            cz, cy, cx = int(rng.integers(2, shape[0]-2)), int(rng.integers(5, shape[1]-5)), int(rng.integers(5, shape[2]-5))
            rz, ry, rx = rng.uniform(1.25, 2.5), rng.uniform(2.0, 3.5), rng.uniform(2.0, 3.5)
            cell = ((z-cz)/rz)**2 + ((y-cy)/ry)**2 + ((x-cx)/rx)**2 <= 1
            cell &= labels == 0
            labels[cell] = obj
            img[0, cell] += rng.uniform(.65, .95)  # nuclear
            img[1, cell] += rng.uniform(.7, .9) if obj % 3 == 0 else .04  # CD8-like
            img[2, cell] += rng.uniform(.7, .9) if obj % 3 != 0 else .05  # panCK-like
        # Synthetic planar vessel marker, not a biologically faithful vasculature model.
        vessel = np.abs(x - (7 + i)) < 2
        img[3] += np.broadcast_to(vessel, shape).astype(np.float32) * .75
        img[0] = gaussian_filter(img[0], (0.5, 0.7, 0.7))
        sid = f"synthetic_{i:02d}"
        filename = f"{sid}.npz"
        np.savez_compressed(out/filename, image=img, mask=labels, channels=np.array(["Nuclear", "CD8", "panCK", "CD31"]))
        split = "train" if i < samples-2 else ("val" if i == samples-2 else "test")
        rows.append(dict(sample_id=sid, patient_id=f"simulated_patient_{i:02d}", split=split,
                         path=filename, spacing_z_um=2., spacing_y_um=.8, spacing_x_um=.8))
    pd.DataFrame(rows).to_csv(out/"manifest.csv", index=False)
    return out/"manifest.csv"


def inspect_manifest(path, out=None):
    df = read_manifest(path)
    stats = []
    for row in df.itertuples():
        image, mask, channels = load_sample(row.path)
        stats.append(dict(sample_id=row.sample_id, split=row.split, patient_id=row.patient_id,
                          shape=list(image.shape), channels=channels, n_reference_instances=int(len(np.unique(mask))-1),
                          labeled_fraction=float(np.mean(mask > 0)),
                          spacing_um=[float(row.spacing_z_um), float(row.spacing_y_um), float(row.spacing_x_um)]))
    result = dict(samples=stats, splits=df.split.value_counts().to_dict(),
                  caveat="Check mask annotation provenance and whether labels are suitable ground truth before training.")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
