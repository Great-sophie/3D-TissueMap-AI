#!/usr/bin/env python3
"""Read-only diagnostic for 3D-TissueMap center/boundary model (VAL only).

Requires existing scripts/train_segmentation.py, scripts/train_center_boundary.py,
and scripts/optimize_instance_separation.py from the project. Never reads test tiles.
Shows whether learned center markers are missing, and whether boundary predictions
separate foreground/edges. Watershed labels are NOT manual ground truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_segmentation import read_data
from train_center_boundary import make_model, predict_heads, guided_watershed, boundary_target, center_target
from optimize_instance_separation import watershed_instances


def compact_stats(x):
    x=np.asarray(x,dtype=np.float32)
    if x.size==0:return {'median': None, 'q90': None}
    return {'median':float(np.median(x)), 'q90':float(np.quantile(x,.90))}


def safe_corr(a,b):
    a,b=np.asarray(a,dtype=np.float32).ravel(),np.asarray(b,dtype=np.float32).ravel()
    if a.size<2 or float(np.std(a))<1e-8 or float(np.std(b))<1e-8: return None
    return float(np.corrcoef(a,b)[0,1])


def center_peaks(center, fg, threshold=.35, min_distance=3):
    fg=np.asarray(fg,dtype=bool)
    if not fg.any(): return np.empty((0,3),dtype=int)
    return peak_local_max(ndi.gaussian_filter(center,(0.5,1.,1.)),
                          min_distance=min_distance, threshold_abs=threshold,
                          labels=fg, exclude_border=False)


def crop_centroid_audit(labels):
    labels=np.asarray(labels)
    ids=np.unique(labels);ids=ids[ids>0]
    if not len(ids):return {'n_instances':0,'centroids_outside_own_ID':0,'fraction_outside':None}
    loc=ndi.center_of_mass(labels>0,labels=labels,index=ids.tolist())
    out=0
    for label, coord in zip(ids,loc):
        if not np.all(np.isfinite(coord)):
            out+=1;continue
        idx=tuple(int(np.clip(np.rint(p),0,d-1)) for p,d in zip(coord,labels.shape))
        out+=int(labels[idx]!=label)
    return {'n_instances':int(len(ids)),'centroids_outside_own_ID':int(out),'fraction_outside':float(out/len(ids))}


def audit(args):
    args.out.mkdir(parents=True, exist_ok=True)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    ckpt=args.run_dir/'center_boundary_3d.pt'
    if not ckpt.is_file():raise FileNotFoundError(ckpt)
    state=torch.load(ckpt,map_location='cpu',weights_only=True)
    if state.get('output_heads') != ['foreground','inner_instance_contour','pseudo_center']:
        raise RuntimeError('Unexpected output head semantics')
    model=make_model(int(state['base']),None)
    model.load_state_dict(state['state_dict'],strict=True)
    model=model.to(device).eval()
    img, ref, _, _ = read_data(args.data_dir)
    splits=pd.read_csv(args.split_run_dir/'tile_splits.csv')
    tiles=splits.loc[splits.split.eq('val')]
    if tiles.empty:raise ValueError('No validation tiles found')
    rows=[]
    for i,t in enumerate(tiles.itertuples(index=False),1):
        y,x,w=int(t.y),int(t.x),int(t.w)
        with torch.inference_mode():
            probs=predict_heads(model,img[:,y:y+w,x:x+w],device)
        border=args.guard
        if border*2>=w:raise ValueError('guard too large')
        sl=(slice(None),slice(border,w-border),slice(border,w-border))
        fg,boundary,center=(p[sl] for p in probs)
        rr=ref[:,y+border:y+w-border,x+border:x+w-border]
        mask=(fg>=args.fg_threshold)
        reference_ids=np.unique(rr)
        reference_ids=reference_ids[reference_ids>0]
        peaks=center_peaks(center,mask,args.center_threshold,args.min_distance)
        selected=set()
        if len(peaks):
            selected=set(np.unique(rr[tuple(peaks.T)]).tolist())-{0}
        guided, guided_stats=guided_watershed(fg,boundary,center,
                  fg_threshold=args.fg_threshold,center_threshold=args.center_threshold,
                  min_distance=args.min_distance,boundary_strength=args.boundary_strength)
        distance, distance_stats=watershed_instances(fg,threshold=args.fg_threshold,
                  min_distance=2,sigma=.8,ensure_each_component=True)
        bound_ref=boundary_target(rr).astype(bool)
        # Boundaries are pseudo-instance contours, INCLUDING cell-background edges.
        b_positive=compact_stats(boundary[bound_ref]);b_negative=compact_stats(boundary[~bound_ref])
        row=dict(y=y,x=x,n_reference=int(len(reference_ids)),
                 n_foreground_voxels=int(mask.sum()),
                 n_center_local_peaks=int(len(peaks)),
                 n_ref_instances_hit_by_center_peak=int(len(selected)),
                 ref_peak_coverage=float(len(selected)/max(1,len(reference_ids))),
                 n_guided_seeds=int(guided_stats['n_seeds']),
                 n_fallback=int(guided_stats['n_fallback']),
                 n_guided_instances=int(len(np.unique(guided))-1),
                 n_distance_seeds=int(distance_stats['n_seeds']),
                 n_distance_instances=int(len(np.unique(distance))-1),
                 center_prob_median_fg=compact_stats(center[mask])['median'],
                 center_prob_q90_fg=compact_stats(center[mask])['q90'],
                 boundary_prob_median_pseudoboundary=b_positive['median'],
                 boundary_prob_median_nonboundary=b_negative['median'],
                 center_boundary_corr=safe_corr(center[mask],boundary[mask]))
        rows.append(row)
        if i==1:
            zz=rr.shape[0]//2
            fig,axes=plt.subplots(2,3,figsize=(12,8))
            contents=[(img[zz,y+border:y+w-border,x+border:x+w-border],'Ir193','gray',None),
                      (rr[zz],'Pseudo-instance labels','nipy_spectral',None),
                      (fg[zz],'Foreground probability','viridis',1),
                      (boundary[zz],'Boundary probability','viridis',1),
                      (center[zz],'Center probability','viridis',1),
                      (center_target(rr)[zz],'Pseudo-center target','viridis',1)]
            for ax,(arr,title,cmap,vmax) in zip(axes.flat,contents):
                ax.imshow(arr,cmap=cmap,vmin=0,vmax=vmax)
                ax.set_title(title,fontsize=10);ax.axis('off')
            fig.suptitle('VAL tile 1 only: shared 0–1 probability scale; NOT manual GT',fontsize=11)
            fig.tight_layout();fig.savefig(args.out/'head_calibration_visual.png',dpi=130);plt.close(fig)
        print(f'VAL {i}/{len(tiles)}: reference={len(reference_ids)} center_peaks={len(peaks)} '
              f'guided_seeds={guided_stats["n_seeds"]} distance_seeds={distance_stats["n_seeds"]}',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(args.out/'head_diagnostics_validation.csv',index=False)
    report={'scope':'Validation-only within one specimen; watershed pseudo-reference, not manual truth',
            'source_checkpoint':str(ckpt),
            'thresholds':{'fg':args.fg_threshold,'center':args.center_threshold,'min_distance_voxels':args.min_distance},
            'medians':{k: (None if not np.issubdtype(df[k].dtype,np.number) else (None if df[k].isna().all() else float(df[k].median()))) for k in df.columns if k not in ['y','x']},
            'warning':'This is a diagnostic, not a model selection experiment; labels, registration and spacing remain uncertain.'}
    (args.out/'head_diagnostics_validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    train=splits.loc[splits.split.eq('train')]
    if not train.empty:
        t=train.iloc[0]
        z0=0; y0=int(t.y)+args.guard;x0=int(t.x)+args.guard
        patch=ref[z0:z0+8,y0:y0+64,x0:x0+64]
        if patch.shape==(8,64,64):
            report['sample_training_crop_centroid_audit']=crop_centroid_audit(patch)
            (args.out/'head_diagnostics_validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    print('Saved:',args.out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--run-dir',type=Path,default=Path('runs/center_boundary_learning'))
    p.add_argument('--out',type=Path,default=Path('runs/center_boundary_head_audit'))
    p.add_argument('--split-run-dir',type=Path,default=Path('runs/segmentation_extended'))
    p.add_argument('--fg-threshold',type=float,default=.5)
    p.add_argument('--center-threshold',type=float,default=.35)
    p.add_argument('--min-distance',type=int,default=3)
    p.add_argument('--boundary-strength',type=float,default=2.0)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args()
    audit(args)

if __name__=='__main__':
    main()
