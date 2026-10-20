#!/usr/bin/env python3
"""Validation-only: 3D connected components vs distance-watershed instance baselines.

Uses the already-trained 3D U-Net and author's watershed instances within the
same specimen. Outputs measure PSEUDO-REFERENCE agreement, NOT cell accuracy,
patient generalization, physical distances, or biological discoveries.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_segmentation import UNet, read_data, predict_tile


def segment(prob, mode='watershed', threshold=.5, min_distance=3):
    fg=np.asarray(prob)>=threshold
    if mode=='components':
        result,_=ndi.label(fg,structure=ndi.generate_binary_structure(3,1))
        return result.astype(np.int32)
    if mode!='watershed': raise ValueError(mode)
    if not np.any(fg):return np.zeros_like(fg,dtype=np.int32)
    dist=ndi.distance_transform_edt(fg) # VOXEL UNITS, spacing unverified
    dist_smooth=ndi.gaussian_filter(dist.astype(np.float32),sigma=0.8)
    coords=peak_local_max(dist_smooth,min_distance=min_distance,threshold_abs=1.0,
                          labels=fg,exclude_border=False)
    marker=np.zeros_like(fg,dtype=np.int32)
    for i,coord in enumerate(coords,1):marker[tuple(coord)]=i
    if not len(coords):
        marker,_=ndi.label(fg,structure=ndi.generate_binary_structure(3,1))
    return watershed(-dist_smooth,markers=marker,mask=fg).astype(np.int32)


def compare_instances(reference,predicted,iou_threshold=.5):
    # IDs are evaluated within a limited XY crop. Border objects are truncated:
    # this screening comparison MUST NOT be interpreted as an unbiased PQ score.
    ref_ids,ref_counts=np.unique(reference[reference>0],return_counts=True)
    pred_ids,pred_counts=np.unique(predicted[predicted>0],return_counts=True)
    nref,npred=len(ref_ids),len(pred_ids)
    if not nref or not npred:
        tp=0;fp=npred;fn=nref
        return dict(n_reference=nref,n_predicted=npred,tp=tp,fp=fp,fn=fn,
                    precision=0.,recall=0.,pq=0.,mean_matched_iou=None)
    both=(reference>0)&(predicted>0)
    if not both.any():
        return dict(n_reference=nref,n_predicted=npred,tp=0,fp=npred,fn=nref,
                    precision=0.,recall=0.,pq=0.,mean_matched_iou=None)
    pairs,intersections=np.unique(np.stack([reference[both],predicted[both]],axis=1),axis=0,return_counts=True)
    ri=np.searchsorted(ref_ids,pairs[:,0]);pi=np.searchsorted(pred_ids,pairs[:,1])
    iou=intersections/(ref_counts[ri]+pred_counts[pi]-intersections)
    matrix=np.zeros((nref,npred),dtype=np.float32)
    matrix[ri,pi]=iou
    # At IoU > .5, no object can match multiple opponents, still use assignment
    # to handle the general case robustly.
    rr,pp=linear_sum_assignment(1-matrix)
    ok=matrix[rr,pp]>=iou_threshold
    vals=matrix[rr[ok],pp[ok]]
    tp=int(ok.sum());fp=npred-tp;fn=nref-tp
    return dict(n_reference=nref,n_predicted=npred,tp=tp,fp=fp,fn=fn,
                precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                pq=float(vals.sum()/(tp+.5*fp+.5*fn)) if tp+fp+fn else 0.,
                mean_matched_iou=float(vals.mean()) if tp else None)


def load_model(path, device):
    # Trust local project checkpoints only.
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('model_dim')!=3:raise ValueError('3D U-Net checkpoint expected')
    model=UNet(dim=3,base=int(ck['base']))
    model.load_state_dict(ck['state_dict'])
    return model.to(device).eval()


def run(data_dir:Path, run_dir:Path, out:Path, device:str, threshold:float, min_distance:int, guard:int):
    out.mkdir(parents=True,exist_ok=True)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    img,ref,indices,_=read_data(data_dir)
    tdf=pd.read_csv(run_dir/'tile_splits.csv')
    val=tdf[tdf.split=='val']
    if len(val)==0:raise ValueError('No validation tiles in split manifest')
    m=load_model(run_dir/'unet_3d.pt',device)
    rows=[]
    for num,t in enumerate(val.itertuples(index=False),1):
        tile=dict(y=int(t.y),x=int(t.x),w=int(t.w))
        if tile['w']<=2*guard:raise ValueError('Invalid guard')
        with torch.inference_mode():pr=predict_tile(m,3,img,tile,device)
        y,x,w=tile['y'],tile['x'],tile['w']
        rr=ref[:,y+guard:y+w-guard,x+guard:x+w-guard]
        pp=pr[:,guard:w-guard,guard:w-guard]
        fg=pp>=threshold
        ds=(2*np.count_nonzero(fg&(rr>0))+1)/(np.count_nonzero(fg)+np.count_nonzero(rr>0)+1)
        variants={}
        for method in ('components','watershed'):
            pred=segment(pp,mode=method,threshold=threshold,min_distance=min_distance)
            res=compare_instances(rr,pred)
            row={'tile_y':y,'tile_x':x,'method':method,'semantic_dice':float(ds),**res}
            rows.append(row); variants[method]=pred
        print(f'val {num}/{len(val)} Y{y} X{x}: semantic Dice={ds:.3f}, components={rows[-2]["n_predicted"]}, watershed={rows[-1]["n_predicted"]}',flush=True)
        if num==1:
            z=img.shape[0]//2
            fig,ax=plt.subplots(1,4,figsize=(14,4))
            extent=(x+guard,x+w-guard,y+w-guard,y+guard)
            ax[0].imshow(img[z,y+guard:y+w-guard,x+guard:x+w-guard],cmap='gray')
            ax[1].imshow(rr[z],cmap='nipy_spectral',vmin=0,interpolation='nearest')
            ax[2].imshow(variants['components'][z],cmap='nipy_spectral',vmin=0,interpolation='nearest')
            ax[3].imshow(variants['watershed'][z],cmap='nipy_spectral',vmin=0,interpolation='nearest')
            for aa,name in zip(ax,['Ir193','Author watershed IDs','Pred connected components','Pred distance-watershed']):
                aa.set_title(name,fontsize=9);aa.axis('off')
            fig.suptitle('Single-specimen validation crop; labels are NOT manual GT',fontsize=11)
            fig.tight_layout();fig.savefig(out/'instance_comparison_validation.png',dpi=150);plt.close(fig)
    df=pd.DataFrame(rows)
    df.to_csv(out/'instance_metrics_validation.csv',index=False)
    summary={k:{'n_validation_tiles':int(len(g)),
                'median_pq':float(g.pq.median()),
                'median_object_precision':float(g.precision.median()),
                'median_object_recall':float(g.recall.median()),
                'median_predicted_objects':float(g.n_predicted.median())}
             for k,g in df.groupby('method')}
    metadata={'scope':'VALIDATION XY tiles in a single specimen only',
              'data_source':'author watershed pseudo-reference, no independent manual instance labels',
              'physical_spacing':'UNKNOWN; voxel-isotropic distance transform is exploratory',
              'evaluation_limitation':'XY crop and Z volume boundaries truncate instances; object metrics boundary-biased and not generalization estimates',
              'threshold':threshold,'min_distance_voxels':min_distance,'guard_pixels':guard,
              'checkpoint':str(run_dir/'unet_3d.pt'), 'results':summary}
    (out/'instance_experiment.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(metadata,indent=2,ensure_ascii=False))
    print('Saved:',out)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--run-dir',type=Path,default=Path('runs/segmentation_extended'))
    p.add_argument('--out',type=Path,default=Path('runs/instance_watershed_validation'))
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--min-distance',type=int,default=3)
    p.add_argument('--guard',type=int,default=12)
    args=p.parse_args()
    if not 0<args.threshold<1: p.error('--threshold must lie between 0 and 1')
    if args.min_distance<1 or args.guard<0:p.error('invalid min-distance or guard')
    run(args.data_dir,args.run_dir,args.out,args.device,args.threshold,args.min_distance,args.guard)

if __name__=='__main__':main()
