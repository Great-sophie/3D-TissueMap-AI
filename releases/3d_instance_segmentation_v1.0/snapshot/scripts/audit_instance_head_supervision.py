#!/usr/bin/env python3
"""TRAIN-only diagnostics of random patch sampling and center/boundary pseudo-targets.

Usage from merged 3D-TissueMap-AI project root:
python scripts/audit_instance_head_supervision.py \
    --data-dir data/inspect/LVILymph \
    --run-dir runs/segmentation_extended \
    --out runs/instance_head_supervision_audit \
    --samples 200 --patch-z 8 --patch-xy 64

Never reads validation/test image pixels or modifies checkpoints. Physical spacing
is unverified; all measurements use voxels. Existing author watershed labels
are pseudo labels, not independent cell ground truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_segmentation import read_data
from train_center_boundary import crop_train, center_target, boundary_target


def centroid_in_own_id(lab: np.ndarray) -> tuple[int, int]:
    ids = np.unique(lab)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return 0, 0
    coords = ndi.center_of_mass(lab > 0, labels=lab, index=ids.tolist())
    outside = 0
    for label, loc in zip(ids, coords):
        if not np.all(np.isfinite(loc)):
            outside += 1
            continue
        voxel = tuple(int(np.clip(np.rint(v), 0, size-1)) for v,size in zip(loc, lab.shape))
        outside += int(lab[voxel] != label)
    return int(outside), int(len(ids))


def sample_patch_stats(img:np.ndarray,lab:np.ndarray) -> dict:
    if img.shape != lab.shape or lab.ndim != 3:
        raise ValueError('Image and label patch must share [Z,Y,X] shape')
    labels = np.unique(lab)
    labels = labels[labels > 0]
    fg = lab > 0
    n_out, n_ids = centroid_in_own_id(lab)
    if len(labels):
        centers = center_target(lab)
        boundaries = boundary_target(lab)
        positive_centers = int((centers > .35).sum())
        center_target_max = float(centers.max())
        boundary_fraction = float(np.mean(boundaries > 0))
    else:
        positive_centers = 0
        center_target_max = 0.
        boundary_fraction = 0.
    # Count instances that intersect patch faces (including z), which can bias
    # cropped-instance centroids. It is not a proof of physical cell truncation.
    clipped = set()
    for side in [lab[0],lab[-1],lab[:,0],lab[:,-1],lab[:,:,0],lab[:,:,-1]]:
        clipped.update(np.unique(side).tolist())
    clipped.discard(0)
    return dict(foreground_fraction=float(fg.mean()),n_instance_ids=int(n_ids),
                n_centroids_outside_own_id=n_out,
                n_instance_ids_touching_crop_face=int(len(clipped)),
                center_target_voxels_gt_035=positive_centers,
                center_target_max=center_target_max,
                boundary_target_fraction=boundary_fraction)


def summary(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),
                p90=float(np.quantile(a,.9)),min=float(a.min()),max=float(a.max()))


def audit(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if args.samples < 1 or args.patch_z < 2 or args.patch_xy < 4:
        raise ValueError('Invalid audit sampling parameters')
    img,lab,_,_ = read_data(args.data_dir)
    audit_path=args.run_dir/'data_audit.json'
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    data_audit=json.loads(audit_path.read_text())
    if data_audit.get('stack_order_check',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise RuntimeError('Image file order was not verified; audit blocked')
    split_path=args.run_dir/'tile_splits.csv'
    if not split_path.exists(): raise FileNotFoundError(split_path)
    all_tiles=pd.read_csv(split_path)
    tiles=all_tiles.loc[all_tiles['split'].eq('train')].to_dict('records')
    if not tiles:raise ValueError('No training tiles')
    rng=np.random.default_rng(args.seed)
    rows=[]
    visual=None
    best_score=-1.
    for n in range(args.samples):
        im_patch,lab_patch=crop_train(img,lab,tiles,rng,args.patch_z,args.patch_xy,args.guard)
        stats=sample_patch_stats(im_patch,lab_patch)
        rows.append(dict(sample=n+1,**stats))
        score=stats['n_instance_ids']+4*stats['foreground_fraction']
        if score > best_score:
            best_score=score
            visual=(im_patch.copy(),lab_patch.copy())
    df=pd.DataFrame(rows)
    df.to_csv(args.out/'train_patch_sampling.csv',index=False)
    fg=df['foreground_fraction'].to_numpy()
    objects=df['n_instance_ids'].to_numpy()
    report=dict(scope='TRAIN patches only; val/test not sampled',
                n_samples=len(df),random_seed=args.seed,
                patch_shape_ZYX=[args.patch_z,args.patch_xy,args.patch_xy],
                empty_patches=int((objects==0).sum()),
                empty_patch_fraction=float(np.mean(objects==0)),
                patches_with_fewer_than_3_ids=int((objects<3).sum()),
                fewer_than_3_fraction=float(np.mean(objects<3)),
                foreground_fraction=summary(fg),instance_id_count=summary(objects),
                fraction_patches_centroid_outside=float(np.mean(df['n_centroids_outside_own_id']>0)),
                crop_face_intersecting_ids=summary(df['n_instance_ids_touching_crop_face']),
                center_target_voxels_gt_035=summary(df['center_target_voxels_gt_035']),
                boundary_target_fraction=summary(df['boundary_target_fraction']),
                caution=['Labels are automated watershed pseudo references, not manual GT',
                         'Crop-face IDs may represent truncated objects but this is not independently verified',
                         'A single random empty crop is not enough to infer sampling failure',
                         'Output-head activation thresholds cannot fix incorrect supervision alone'])
    (args.out/'train_patch_sampling_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if visual is not None:
        im,labels=visual
        z=im.shape[0]//2
        ctr=center_target(labels)
        bound=boundary_target(labels)
        fig,axs=plt.subplots(1,4,figsize=(12,3.5))
        for ax,arr,title,cmap in zip(axs,[im[z],labels[z],ctr[z],bound[z]],
                       ['Ir193 (TRAIN patch)','Pseudo instance IDs','Cropped centroid target','Pseudo boundary target'],
                       ['gray','nipy_spectral','viridis','gray']):
            ax.imshow(arr,cmap=cmap,vmin=0,vmax=1 if 'target' in title else None)
            ax.set_title(title,fontsize=9);ax.axis('off')
        fig.suptitle('TRAIN-only supervision audit — all units in voxels; targets are pseudo labels',fontsize=10)
        fig.tight_layout();fig.savefig(args.out/'training_targets_example.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
    print('Saved:', args.out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--run-dir',type=Path,default=Path('runs/segmentation_extended'))
    p.add_argument('--out',type=Path,default=Path('runs/instance_head_supervision_audit'))
    p.add_argument('--samples',type=int,default=200)
    p.add_argument('--patch-z',type=int,default=8)
    p.add_argument('--patch-xy',type=int,default=64)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--seed',type=int,default=42)
    audit(p.parse_args())

if __name__=='__main__':main()
