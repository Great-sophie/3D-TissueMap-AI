#!/usr/bin/env python3
"""Exploratory supervoxel-to-region affinity aggregation (VAL only).

This is deliberately a diagnostic, NOT a validated instance decoder.
Blockwise supervoxels prevent single-voxel fragmentation; region-interface
aggregation reduces sensitivity to isolated voxel-edge errors. A protective
size cap can artificially split large cells, so all scores are preliminary.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_segmentation import read_data
from train_voxel_affinity import make_network, predict_prob
from instance_watershed_baseline import segment
from evaluate_affinity_graph_pilot import sparse_pq, component_stats


def block_supervoxels(fg: np.ndarray, block_zyx=(2,6,6)) -> np.ndarray:
    fg = np.asarray(fg, bool)
    if fg.ndim != 3 or any(s < 1 for s in block_zyx):
        raise ValueError('fg must be 3D and all block dimensions > 0')
    labels = np.zeros(fg.shape, np.int32)
    current = 0
    structure = ndi.generate_binary_structure(3, 1)
    for z in range(0,fg.shape[0],block_zyx[0]):
        for y in range(0,fg.shape[1],block_zyx[1]):
            for x in range(0,fg.shape[2],block_zyx[2]):
                slices = (slice(z,z+block_zyx[0]),slice(y,y+block_zyx[1]),slice(x,x+block_zyx[2]))
                crop = fg[slices]
                if not crop.any():continue
                lab,n = ndi.label(crop,structure=structure)
                if n:
                    dst=labels[slices]
                    dst[lab>0]=lab[lab>0]+current
                    current += n
    return labels


def interface_edges(sv:np.ndarray, p_same:np.ndarray,z_weight:float=0.25):
    """Return (a,b,mean logit,negative share,nfaces) for contacting supervoxels."""
    if sv.ndim!=3 or p_same.shape!=(3,*sv.shape):
        raise ValueError('expected labels[Z,Y,X] and P(same)[3,Z,Y,X]')
    if z_weight<0:raise ValueError('z_weight must be >=0')
    keys=[]; logs=[]; weights=[]; negative=[]
    codebase=int(sv.max())+1
    for axis in range(3):
        if axis==0 and z_weight==0:continue
        lo=[slice(None)]*3;hi=[slice(None)]*3
        lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        a=sv[tuple(lo)];b=sv[tuple(hi)]
        m=(a>0)&(b>0)&(a!=b)
        if not np.any(m):continue
        a=a[m].astype(np.int64);b=b[m].astype(np.int64)
        k=np.minimum(a,b)*codebase+np.maximum(a,b)
        p=np.clip(p_same[axis][tuple(lo)][m].astype(np.float64),1e-5,1-1e-5)
        keys.append(k);logs.append(np.log(p/(1-p)))
        weights.append(np.full(p.shape,z_weight if axis==0 else 1.,np.float64))
        negative.append((p<0.5).astype(np.float64))
    if not keys:
        return np.empty((0,5),np.float64)
    key=np.concatenate(keys);log=np.concatenate(logs);w=np.concatenate(weights);neg=np.concatenate(negative)
    unique,inverse=np.unique(key,return_inverse=True)
    ws=np.bincount(inverse,weights=w,minlength=len(unique))
    score=np.bincount(inverse,weights=w*log,minlength=len(unique))/np.maximum(ws,1e-8)
    negfrac=np.bincount(inverse,weights=w*neg,minlength=len(unique))/np.maximum(ws,1e-8)
    counts=np.bincount(inverse,minlength=len(unique))
    return np.stack((unique//codebase,unique%codebase,score,negfrac,counts),axis=1)


def merge_regions(sv:np.ndarray,interfaces:np.ndarray,score_threshold=0.2,
                  max_negative_fraction=0.35,max_component_voxels=1200,min_faces=2):
    if max_component_voxels<1:raise ValueError('max_component_voxels must be >=1')
    n=int(sv.max())
    sizes=np.bincount(sv.ravel(),minlength=n+1).astype(np.int64)
    parent=np.arange(n+1,dtype=np.int32)
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=int(parent[a])
        return a
    count_accepted=0
    if len(interfaces):
        candidate=interfaces[(interfaces[:,2]>=score_threshold)&
                             (interfaces[:,3]<=max_negative_fraction)&
                             (interfaces[:,4]>=min_faces)]
        for row in candidate[np.argsort(-candidate[:,2])]:
            a,b=map(int,row[:2]);ra,rb=find(a),find(b)
            if ra==rb or sizes[ra]+sizes[rb]>max_component_voxels:continue
            if sizes[ra]<sizes[rb]:ra,rb=rb,ra
            parent[rb]=ra;sizes[ra]+=sizes[rb];count_accepted+=1
    remap=np.zeros(n+1,np.int32)
    labels=np.unique(sv[sv>0])
    root=np.array([find(int(i)) for i in labels],np.int32)
    _,inv=np.unique(root,return_inverse=True)
    remap[labels]=inv+1
    return remap[sv],count_accepted


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((args.split_run_dir/'data_audit.json').read_text())
    if audit.get('stack_order_check',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise RuntimeError('Stack-order check was not passed')
    if not args.accept_provisional_registration:
        raise RuntimeError('Acknowledge unverified registration with --accept-provisional-registration')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    img,ref,_,_=read_data(args.data_dir)
    val=pd.read_csv(args.split_run_dir/'tile_splits.csv')
    val=val[val.split=='val'].head(args.max_tiles)
    if val.empty:raise RuntimeError('no validation tiles')
    ck=torch.load(args.affinity_run_dir/'xyz_downweighted.pt',map_location='cpu',weights_only=True)
    if ck.get('z_weight',0)<=0:raise RuntimeError('Need XYZ-supervised checkpoint')
    model=make_network(int(ck['base']),args.split_run_dir/'unet_3d.pt')
    model.load_state_dict(ck['state_dict'],strict=True)
    model=model.to(device).eval()
    rows=[]
    for t in val.itertuples(index=False):
        y,x,w=int(t.y),int(t.x),int(t.w)
        if w<=2*args.guard:raise ValueError('guard too large')
        with torch.inference_mode():
            p=predict_prob(model,img[:,y:y+w,x:x+w],device)
        g=args.guard
        p=p[:,:,g:w-g,g:w-g]
        ref_crop=ref[:,y+g:y+w-g,x+g:x+w-g]
        fg=p[0]>=0.5
        sv=block_supervoxels(fg,(args.region_z,args.region_xy,args.region_xy))
        interfaces=interface_edges(sv,p[1:],args.z_weight)
        eligible = interfaces[(interfaces[:,2]>=args.score_threshold)&(interfaces[:,3]<=args.max_negative_fraction)&(interfaces[:,4]>=args.min_faces)]
        pred,n_merged=merge_regions(sv,interfaces,args.score_threshold,
                                    args.max_negative_fraction,args.max_component_voxels,args.min_faces)
        ws=segment(p[0],mode='watershed',threshold=0.5,min_distance=3)
        fig,axs=plt.subplots(1,4,figsize=(15,4))
        z=int(p.shape[1]//2)
        axs[0].imshow(img[z,y+g:y+w-g,x+g:x+w-g],cmap='gray')
        for ax,title,array in zip(axs[1:],['Author pseudo IDs','Distance watershed','Region graph'],[ref_crop,ws,pred]):
            slab=array[z].astype(np.int32)
            ax.imshow(np.ma.masked_equal(slab,0),cmap='nipy_spectral',interpolation='nearest')
        for ax,title in zip(axs,['Ir193','Author pseudo IDs','Distance watershed','Region graph']):
            ax.set_title(title);ax.axis('off')
        fig.suptitle('VAL only / author pseudo-labels / diagnostic region graph')
        fig.tight_layout();fig.savefig(args.out/f'region_graph_comparison_{y}_{x}.png',dpi=150)
        plt.close(fig)
        for method,labels in [('distance_watershed',ws),('region_affinity_merging',pred)]:
            rows.append(dict(method=method,tile_y=y,tile_x=x,
                             **sparse_pq(ref_crop,labels),**component_stats(labels),
                             n_supervoxels=int(sv.max()),n_interfaces=len(interfaces),
                             n_candidate_interfaces=int(len(eligible)),n_merges=n_merged))
        print(f'VAL ({y},{x}) supervoxels={sv.max()} interfaces={len(interfaces)} candidates={len(eligible)} merges={n_merged}; '
              f'Watershed PQ={rows[-2]["pq"]:.4f}; Region-graph PQ={rows[-1]["pq"]:.4f}',flush=True)
    pd.DataFrame(rows).to_csv(args.out/'region_graph_validation.csv',index=False)
    result={'scope':'Single-specimen, repeatedly reused VAL only; test not touched',
            'reference':'Author-produced watershed pseudo-labels, not independently annotated cells',
            'decoder':'Small grid-connected supervoxels + pooled region-interface log-odds + greedy size-capped merging; NOT Mutex Watershed',
            'params':{'block_zyx':[args.region_z,args.region_xy,args.region_xy],
                      'score_threshold_logit':args.score_threshold,'max_negative_fraction':args.max_negative_fraction,
                      'z_weight':args.z_weight,'max_component_voxels':args.max_component_voxels,
                      'min_interface_faces':args.min_faces},
            'rows':rows,
            'limitations':['Grid supervoxels are a computational diagnostic and not anatomical units',
                           'Volume cap can under-reconstruct large cells',
                           'Interface scores are not independent probabilities',
                           'Registration and physical voxel spacing unverified',
                           'No threshold calibration or independent test validation']}
    (args.out/'region_graph_pilot.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('Saved:',args.out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--split-run-dir',type=Path,required=True)
    p.add_argument('--affinity-run-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-tiles',type=int,default=1)
    p.add_argument('--region-z',type=int,default=2)
    p.add_argument('--region-xy',type=int,default=6)
    p.add_argument('--z-weight',type=float,default=0.25)
    p.add_argument('--score-threshold',type=float,default=0.2,help='mean edge log odds; 0.2 approximately P(same)=0.55')
    p.add_argument('--max-negative-fraction',type=float,default=0.35)
    p.add_argument('--max-component-voxels',type=int,default=1200)
    p.add_argument('--min-faces',type=int,default=2)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--accept-provisional-registration',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
