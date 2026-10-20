#!/usr/bin/env python3
"""Diagnose 2D/3D semantic-segmentation outputs without retraining.

Reads example_predictions.npz from the 3D-TissueMap AI exploratory single-tissue
IMC experiment. Works at the fixed 0.50 decision threshold. This is a descriptive
held-out-tile diagnostic, NOT threshold tuning or independent generalization.

Run from project root:
    python scripts/segmentation_diagnostics.py \
       --input runs/segmentation_baselines/example_predictions.npz \
       --out runs/segmentation_diagnostics
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

THRESHOLD = 0.5


def score(prob, truth, threshold=THRESHOLD):
    pred = prob >= threshold
    pos = truth > 0
    tp = int((pred & pos).sum())
    fp = int((pred & ~pos).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & ~pos).sum())
    bg = prob[~pos]
    fg = prob[pos]
    safe = lambda n, d: float(n / d) if d else None
    return {
        'threshold': threshold,
        'voxels': int(pos.size),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'reference_foreground_fraction': float(pos.mean()),
        'predicted_foreground_fraction': float(pred.mean()),
        'dice': safe(2*tp, 2*tp+fp+fn),
        'iou': safe(tp, tp+fp+fn),
        'precision': safe(tp, tp+fp),
        'recall': safe(tp, tp+fn),
        'background_false_positive_fraction': safe(fp, fp+tn),
        'foreground_false_negative_fraction': safe(fn, fn+tp),
        'background_probability_median': float(np.median(bg)) if bg.size else None,
        'foreground_probability_median': float(np.median(fg)) if fg.size else None,
    }


def error_rgb(prob, truth):
    pred = prob >= THRESHOLD
    pos = truth > 0
    rgb = np.zeros((*pos.shape,3),dtype=np.uint8)
    rgb[pred & pos] = (58,190,90)    # TP green
    rgb[pred & ~pos] = (230,65,65)   # FP red
    rgb[~pred & pos] = (85,140,240) # FN blue
    return rgb


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('runs/segmentation_baselines/example_predictions.npz'))
    ap.add_argument('--out',type=Path,default=Path('runs/segmentation_diagnostics'))
    args=ap.parse_args()
    if not args.input.exists():
        ap.error(f'Input not found: {args.input} ; run the segmentation training first')
    args.out.mkdir(parents=True,exist_ok=True)
    with np.load(args.input,allow_pickle=False) as a:
        required={'reference_binary','p2d','p3d'}
        if not required.issubset(set(a.files)):
            raise ValueError(f'Expected keys {sorted(required)}; found {a.files}')
        ref=np.asarray(a['reference_binary'])>0
        p2=np.asarray(a['p2d'],dtype=np.float32)
        p3=np.asarray(a['p3d'],dtype=np.float32)
    if not (ref.ndim==3 and ref.shape==p2.shape==p3.shape):
        raise ValueError(f'Expected matching ZYX arrays: {ref.shape} {p2.shape} {p3.shape}')
    if not (np.isfinite(p2).all() and np.isfinite(p3).all()):
        raise ValueError('Nonfinite model probabilities found')
    if p2.min()<0 or p3.min()<0 or p2.max()>1 or p3.max()>1:
        raise ValueError('Predictions are not bounded probabilities 0..1')
    results={'source':str(args.input), 'reference':'author-provided watershed pseudo-labels; NOT manual GT',
             'scope':'one held-out XY tile in one specimen; NOT independent validation',
             'purpose':'fixed 0.50 threshold diagnostic only; do NOT select a threshold on test outputs',
             '2d':score(p2,ref),'3d':score(p3,ref)}
    (args.out/'diagnostics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    z=ref.shape[0]//2
    fig,axs=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    axs[0,0].imshow(ref[z],cmap='gray',vmin=0,vmax=1)
    axs[0,0].set_title('Watershed reference (NOT manual GT)')
    c0=axs[0,1].imshow(p2[z],cmap='viridis',vmin=0,vmax=1)
    axs[0,1].set_title('2D U-Net probabilities (0..1)')
    axs[0,2].imshow(p3[z],cmap='viridis',vmin=0,vmax=1)
    axs[0,2].set_title('3D U-Net probabilities (0..1)')
    axs[1,0].imshow(error_rgb(p2[z],ref[z]));axs[1,0].set_title('2D errors @ threshold 0.5')
    axs[1,1].imshow(error_rgb(p3[z],ref[z]));axs[1,1].set_title('3D errors @ threshold 0.5')
    rng=np.random.default_rng(42)
    for name,p in [('2D',p2),('3D',p3)]:
        bg=p[~ref]
        if bg.size>50000: bg=bg[rng.choice(bg.size,50000,replace=False)]
        axs[1,2].hist(bg,bins=np.linspace(0,1,41),alpha=.55,density=True,label=name)
    axs[1,2].axvline(.5,color='k',linestyle='--',label='Fixed threshold')
    axs[1,2].set(xlabel='Probability on watershed-negative voxels',ylabel='Density',title='Background probability distribution')
    axs[1,2].legend()
    legend=[Patch(facecolor=(58/255,190/255,90/255),label='True positive'),
            Patch(facecolor=(230/255,65/255,65/255),label='False positive'),
            Patch(facecolor=(85/255,140/255,240/255),label='False negative')]
    axs[1,0].legend(handles=legend,loc='upper right',fontsize=7,framealpha=.85)
    for ax in axs.flat[:5]: ax.set_axis_off()
    fig.suptitle('3D-TissueMap AI | same-specimen exploratory error diagnosis',fontsize=14)
    fig.savefig(args.out/'prediction_failure_modes.png',dpi=160)
    plt.close(fig)
    for name in ('2d','3d'):
        s=results[name]
        print(f"{name.upper()}: dice={s['dice']:.4f}, iou={s['iou']:.4f}, precision={s['precision']}, recall={s['recall']}, "
              f"pred_fg={s['predicted_foreground_fraction']:.3f}, ref_fg={s['reference_foreground_fraction']:.3f}, "
              f"bg_FP={s['background_false_positive_fraction']:.3f}, bg_prob_median={s['background_probability_median']:.3f}")
    print('Saved:',args.out/'diagnostics.json',args.out/'prediction_failure_modes.png')
    print('CAUTION: Do not tune decision thresholds using this held-out test tile.')

if __name__=='__main__':main()
