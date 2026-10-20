#!/usr/bin/env python3
"""3D-TissueMap AI | fixed training patch memorization diagnostic.

Purpose: verify that a small 3D U-Net can learn a single pseudo-labelled
training patch before increasing training length or changing inference.

This is an optimization sanity check, NOT a validation/test score, cell-instance
metric, evidence of generalization, or accuracy against manual ground truth.

Run from project root after placing this file in scripts/:
 python scripts/segmentation_fit_check.py \
   --data-dir data/inspect/LVILymph \
   --run-dir runs/segmentation_baselines \
   --out runs/segmentation_fit_check \
   --steps 120 --patch-z 8 --patch-xy 64 --base 4
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_segmentation import UNet, loss_fn, read_data


def select_training_patch(image, labels, split_table, pz=8, pxy=64, guard=12,
                          trials=200, seed=42):
    """Sample only from the original training tiles, preferring mixed labels."""
    candidates = split_table.loc[split_table['split'] == 'train']
    if candidates.empty:
        raise ValueError('No training tiles; refusing to sample validation/test')
    Z, H, W = image.shape
    if pz < 4 or pz % 2 or pz > Z or pxy < 16 or pxy % 2:
        raise ValueError('Patch Z/XY must be even, Z in [4,depth], XY >= 16')
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        tile = candidates.iloc[int(rng.integers(len(candidates)))]
        y0, x0, n = int(tile['y']), int(tile['x']), int(tile['w'])
        if n - 2*guard < pxy:
            raise ValueError('patch-xy cannot fit in training tile after guard')
        z = int(rng.integers(0, Z-pz+1))
        y = int(rng.integers(y0+guard, y0+n-guard-pxy+1))
        x = int(rng.integers(x0+guard, x0+n-guard-pxy+1))
        crop_y = labels[z:z+pz, y:y+pxy, x:x+pxy] > 0
        frac = float(crop_y.mean())
        # Prefer nontrivial mixed samples, not all background or all foreground.
        if 0.05 <= frac <= 0.80:
            score = abs(frac-0.30)
            if best is None or score < best[0]:
                best = (score,z,y,x,frac,dict(tile))
    if best is None:
        raise RuntimeError('No mixed-foreground training patch in attempts; inspect mask and train tiles')
    _,z,y,x,frac,tile = best
    im = image[z:z+pz,y:y+pxy,x:x+pxy].astype(np.float32,copy=True)
    ref = (labels[z:z+pz,y:y+pxy,x:x+pxy] > 0).astype(np.float32,copy=True)
    if im.shape != (pz,pxy,pxy):
        raise RuntimeError(f'Unexpected patch shape {im.shape}')
    return im, ref, {'z':z,'y':y,'x':x,'patch_shape_ZYX':list(im.shape),
                     'reference_foreground':frac,'source_split':'train',
                     'source_tile_y':int(tile['y']),'source_tile_x':int(tile['x'])}


def summarize(prob, truth):
    y = truth > 0.5
    p = prob >= 0.5
    tp = int((p & y).sum()); fp = int((p & ~y).sum())
    fn = int((~p & y).sum()); tn = int((~p & ~y).sum())
    return {
        'dice': float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else None,
        'precision': float(tp/(tp+fp)) if (tp+fp) else 0.,
        'recall': float(tp/(tp+fn)) if (tp+fn) else 0.,
        'reference_foreground': float(y.mean()),
        'predicted_foreground': float(p.mean()),
        'background_FP_rate': float(fp/(fp+tn)) if (fp+tn) else None,
        'median_prob_on_background': float(np.median(prob[~y])) if (~y).any() else None,
        'median_prob_on_foreground': float(np.median(prob[y])) if y.any() else None,
    }


@torch.inference_mode()
def predict(model, x):
    model.eval()
    return torch.sigmoid(model(x))[0,0].detach().cpu().numpy()


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.steps < 1 or args.trials < 1:
        raise ValueError('steps/trials must be positive')
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device
    if device == 'auto': device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    args.out.mkdir(parents=True,exist_ok=True)
    split_file = args.run_dir / 'tile_splits.csv'
    if not split_file.exists():
        raise FileNotFoundError(f'Missing {split_file}; run data audit first')
    splits = pd.read_csv(split_file)
    image,mask,_,_ = read_data(args.data_dir)
    im,ref,provenance = select_training_patch(image,mask,splits,
        args.patch_z,args.patch_xy,args.guard,args.trials,args.seed)
    model=UNet(dim=3,base=args.base).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    x=torch.from_numpy(im)[None,None].to(device)
    y=torch.from_numpy(ref)[None,None].to(device)
    start_prob=predict(model,x)
    rows=[]
    for step in range(1,args.steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits=model(x)
        loss=loss_fn(logits,y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at step {step}')
        loss.backward()
        total_grad=sum(float(p.grad.detach().abs().sum()) for p in model.parameters() if p.grad is not None)
        if not np.isfinite(total_grad) or total_grad == 0:
            raise FloatingPointError('Zero/non-finite gradient; inspect graph, normalization and labels')
        optimizer.step()
        if step == 1 or step%args.log_every == 0 or step == args.steps:
            prob=predict(model,x)
            stat=summarize(prob,ref)
            rows.append({'step':step,'loss_before_update':float(loss.item()), **stat})
            print(f"step={step:4d} loss={loss.item():.4f} dice={stat['dice']:.4f} "
                  f"pred_fg={stat['predicted_foreground']:.3f} bg_FP={stat['background_FP_rate']:.3f}",flush=True)
    last_prob=predict(model,x)
    initial=summarize(start_prob,ref)
    final=summarize(last_prob,ref)
    pd.DataFrame(rows).to_csv(args.out/'train_patch_history.csv',index=False)
    np.savez_compressed(args.out/'train_patch_predictions.npz',image=im,reference=ref.astype('uint8'),
                        initial_prob=start_prob.astype('float16'),final_prob=last_prob.astype('float16'))
    z=im.shape[0]//2
    fig,ax=plt.subplots(1,4,figsize=(13,3.5))
    for a,img,title,vmin,vmax,cmap in [
        (ax[0],im[z],'Ir193 input',None,None,'gray'),
        (ax[1],ref[z],'Watershed pseudo-label',0,1,'gray'),
        (ax[2],start_prob[z],'Before training',0,1,'viridis'),
        (ax[3],last_prob[z],'After training',0,1,'viridis')]:
        a.imshow(img,cmap=cmap,vmin=vmin,vmax=vmax)
        a.set_title(title);a.axis('off')
    fig.suptitle('TRAINING PATCH ONLY: optimization sanity check; NOT model validation',fontsize=10)
    fig.tight_layout();fig.savefig(args.out/'single_patch_learning.png',dpi=150);plt.close(fig)
    result={
        'purpose':'fixed one-patch overfit / optimizer diagnostic (TRAIN split only)',
        'data':'real 3D IMC image with author watershed pseudo-label; one specimen',
        'does_not_establish':'3D instance accuracy, patient generalization, independent segmentation ground truth',
        'device':device,'steps':args.steps,'learning_rate':args.lr,'base':args.base,
        'training_patch':provenance,'before':initial,'after':final,
        'interpretation_hint':'If training-patch Dice rises strongly and bg_FP falls, the network and optimizer can fit this patch; then run longer separated-tile train/validation. If it cannot fit, debug input-label correspondence, optimization and supervision before large runs.'
    }
    (args.out/'fit_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('Files saved:',args.out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--run-dir',type=Path,default=Path('runs/segmentation_baselines'))
    p.add_argument('--out',type=Path,default=Path('runs/segmentation_fit_check'))
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--patch-z',type=int,default=8)
    p.add_argument('--patch-xy',type=int,default=64)
    p.add_argument('--base',type=int,default=4)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--trials',type=int,default=200)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    run(p.parse_args())

if __name__=='__main__': main()
