#!/usr/bin/env python3
"""Diagnostic of supervoxel adjacency and reference-guided merge feasibility.

READS TRAIN spatial tiles only. Reference IDs are used to DIAGNOSE graph topology;
they never train the affinity model, and oracle output is not deployable performance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from train_segmentation import read_data
from train_voxel_affinity import make_network, predict_prob
from evaluate_affinity_region_graph import block_supervoxels, interface_edges
from evaluate_affinity_graph_pilot import sparse_pq


def majority_reference_per_supervoxel(sv: np.ndarray, ref: np.ndarray):
    """Reference majority among positive reference voxels; 0 means no reference overlap.

    Return majority reference ID and purity over ALL voxels of each SV, plus overlap
    purity conditional on reference-positive voxels. Does not read TEST data.
    """
    if sv.shape != ref.shape or sv.ndim != 3:
        raise ValueError('SV and reference must be same 3D shape')
    n=int(sv.max())
    majority=np.zeros(n+1,np.int64)
    purity_all=np.zeros(n+1,np.float64)
    purity_overlap=np.zeros(n+1,np.float64)
    total=np.bincount(sv.ravel().astype(np.int64),minlength=n+1)
    good=(sv>0)&(ref>0)
    if not np.any(good):
        return majority,purity_all,purity_overlap,total
    nodes=sv[good].astype(np.int64)
    refs=ref[good].astype(np.int64)
    # Compress sparse (node, reference) pairs, avoiding a giant dense contingency table.
    pairs=np.column_stack((nodes,refs))
    uniq,count=np.unique(pairs,axis=0,return_counts=True)
    # Sort by node ascending, counts descending, reference ID ascending for deterministic ties.
    order=np.lexsort((uniq[:,1],-count,uniq[:,0]))
    ranked=uniq[order]
    ranked_count=count[order]
    _,first=np.unique(ranked[:,0],return_index=True)
    chosen_nodes=ranked[first,0]
    majority[chosen_nodes]=ranked[first,1]
    chosen_counts=ranked_count[first]
    overlap=np.bincount(nodes,minlength=n+1)
    purity_all[chosen_nodes]=chosen_counts/np.maximum(total[chosen_nodes],1)
    purity_overlap[chosen_nodes]=chosen_counts/np.maximum(overlap[chosen_nodes],1)
    return majority,purity_all,purity_overlap,total


def graph_masks(interfaces: np.ndarray,score_threshold=.2, negative_threshold=.35,min_faces=2):
    if interfaces.ndim!=2 or interfaces.shape[1]!=5:
        raise ValueError('Expected interfaces shape N x 5')
    score=interfaces[:,2]>=score_threshold
    neg=interfaces[:,3]<=negative_threshold
    faces=interfaces[:,4]>=min_faces
    return {'all':np.ones(len(interfaces),dtype=bool), 'score_only':score,
            'negative_only':neg, 'faces_only':faces,
            'all_three':score&neg&faces}


def oracle_connected_output(sv:np.ndarray,edges:np.ndarray,majority:np.ndarray,keep:np.ndarray):
    """Union only direct interfaces whose two nodes share the same positive ref-majority ID.

    This is an optimistic oracle because union decisions use reference labels. It
    intentionally does not teleport noncontacting nodes into the same component.
    """
    n=int(sv.max())
    if len(majority)<=n or len(edges)!=len(keep):
        raise ValueError('Invalid majority map / keep mask')
    parent=np.arange(n+1,dtype=np.int32)
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=int(parent[a])
        return a
    accepted=0
    for e in edges[keep]:
        a,b=int(e[0]),int(e[1])
        if majority[a]<=0 or majority[a]!=majority[b]:
            continue
        a=find(a); b=find(b)
        if a!=b:
            parent[b]=a
            accepted+=1
    if n==0:
        return np.zeros_like(sv,dtype=np.int32),0
    mapping=np.zeros(n+1,dtype=np.int32)
    root=np.fromiter((find(i) for i in range(1,n+1)),dtype=np.int32,count=n)
    _,ind=np.unique(root,return_inverse=True)
    mapping[1:]=ind+1
    return mapping[sv],accepted


def inspect_tile(prob,reference,block_zyx=(1,4,4),foreground_threshold=.5,
                 score_threshold=.2,negative_threshold=.35,min_faces=2,z_weight=.25):
    fg=prob[0]>=foreground_threshold
    sv=block_supervoxels(fg,block_zyx)
    edges=interface_edges(sv,prob[1:],z_weight)
    majority,purity_all,purity_overlap,sv_sizes=majority_reference_per_supervoxel(sv,reference)
    n=int(sv.max())
    support=(majority[1:]>0)
    total_pred=int(fg.sum())
    ref_fg=reference>0
    overlap=int((fg&ref_fg).sum())
    fg_components=int(ndi.label(fg,structure=ndi.generate_binary_structure(3,1))[1])
    fg_metrics={
        'n_predicted_fg_voxels':total_pred,
        'n_reference_fg_voxels':int(ref_fg.sum()),
        'intersection_fg_voxels':overlap,
        'fg_precision':overlap/max(total_pred,1),
        'fg_recall':overlap/max(int(ref_fg.sum()),1),
        'fg_dice':2*overlap/max(total_pred+int(ref_fg.sum()),1),
        'foreground_6_connected_components':fg_components,
        'n_supervoxels':n,
        'n_supervoxels_with_reference_overlap':int(support.sum()),
        'n_supervoxels_without_reference_overlap':int((~support).sum()),
        'fraction_pred_fg_voxels_correct_majority_id':float(np.sum(purity_all[1:]*sv_sizes[1:])/max(total_pred,1)),
        'fraction_overlap_fg_voxels_correct_majority_id':float(np.sum(purity_all[1:]*sv_sizes[1:])/max(overlap,1)),
        'median_supervoxel_overlap_purity':float(np.median(purity_overlap[1:][support])) if np.any(support) else 0.,
    }
    # Unconstrained majority-reference oracle (can reuse one ID across disconnected areas).
    unconstrained=majority[sv].astype(np.int32)
    fg_metrics['oracle_majority_label_assembly_pq_unconstrained']=sparse_pq(reference,unconstrained)
    a=edges[:,0].astype(int) if len(edges) else np.array([],dtype=int)
    b=edges[:,1].astype(int) if len(edges) else np.array([],dtype=int)
    valid=(majority[a]>0)&(majority[b]>0)
    true_same=valid&(majority[a]==majority[b])
    true_diff=valid&(~true_same)
    unknown=~valid
    high_purity=valid&(purity_overlap[a]>=.8)&(purity_overlap[b]>=.8)
    masks=graph_masks(edges,score_threshold,negative_threshold,min_faces)
    audit=[]
    for name,keep in masks.items():
        pred,merges=oracle_connected_output(sv,edges,majority,keep)
        known_kept=keep&valid
        true_same_kept=keep&true_same
        high_same=high_purity&true_same
        audit.append({
            'filter':name,
            'interfaces_total':int(len(edges)),
            'interfaces_kept':int(keep.sum()),
            'oracle_same_edges_total':int(true_same.sum()),
            'oracle_different_edges_total':int(true_diff.sum()),
            'oracle_unknown_edges_total':int(unknown.sum()),
            'oracle_same_edges_kept':int(true_same_kept.sum()),
            'oracle_different_edges_kept':int((keep&true_diff).sum()),
            'oracle_unknown_edges_kept':int((keep&unknown).sum()),
            'majority_same_edge_recall':float(true_same_kept.sum()/max(int(true_same.sum()),1)),
            'kept_majority_same_precision_known_edges':float(true_same_kept.sum()/max(int(known_kept.sum()),1)),
            'high_purity_majority_same_edge_recall':float((keep&high_same).sum()/max(int(high_same.sum()),1)),
            'reference_guided_oracle_merges':int(merges),
            'reference_guided_oracle_connected':sparse_pq(reference,pred),
        })
    return fg_metrics,audit


def run(args):
    if not args.accept_provisional_registration:
        raise RuntimeError('Pass --accept-provisional-registration to acknowledge unverified registration')
    audit=json.loads((args.split_run_dir/'data_audit.json').read_text())
    if audit.get('stack_order_check',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise RuntimeError('Stack order check not verified')
    if args.max_tiles<1 or min(args.region_z,args.region_xy)<1:
        raise ValueError('Need positive tile count and block sizes')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    img,reference,_,_=read_data(args.data_dir)
    tiles=pd.read_csv(args.split_run_dir/'tile_splits.csv')
    tiles=tiles[tiles.split=='train'].head(args.max_tiles)
    if tiles.empty:
        raise RuntimeError('No TRAIN tiles found')
    ck=torch.load(args.affinity_run_dir/'xyz_downweighted.pt',map_location='cpu',weights_only=True)
    if float(ck.get('z_weight',0))<=0:
        raise RuntimeError('XYZ checkpoint needed')
    net=make_network(int(ck['base']),args.split_run_dir/'unet_3d.pt')
    net.load_state_dict(ck['state_dict'],strict=True)
    net=net.to(device).eval()
    args.out.mkdir(parents=True,exist_ok=True)
    results=[];table=[]
    for tile in tiles.itertuples(index=False):
        y,x,w=int(tile.y),int(tile.x),int(tile.w)
        g=args.guard
        if g<0 or 2*g>=w:raise ValueError('Invalid guard')
        with torch.inference_mode():
            p=predict_prob(net,img[:,y:y+w,x:x+w],device)
        p=p[:,:,g:w-g,g:w-g]
        ref=reference[:,y+g:y+w-g,x+g:x+w-g]
        metrics,rows=inspect_tile(p,ref,
              (args.region_z,args.region_xy,args.region_xy),args.foreground_threshold,
              args.score_threshold,args.max_negative_fraction,args.min_faces,args.z_weight)
        results.append({'tile_y':y,'tile_x':x,'metrics':metrics,'candidate_filters':rows})
        for r in rows:
            q=r['reference_guided_oracle_connected']
            table.append({'tile_y':y,'tile_x':x,**{k:v for k,v in r.items() if k!='reference_guided_oracle_connected'},
                          'oracle_connected_pq':q['pq'],'oracle_connected_n_pred':q['n_predicted'],
                          'oracle_connected_recall':q['recall']})
        print(f'TRAIN ({y},{x}) foreground Dice={metrics["fg_dice"]:.4f}; '
              f'FG components={metrics["foreground_6_connected_components"]}; '
              f'SV={metrics["n_supervoxels"]}; oracle PQ all={rows[0]["reference_guided_oracle_connected"]["pq"]:.4f} '
              f'filtered={rows[-1]["reference_guided_oracle_connected"]["pq"]:.4f}',flush=True)
    out={
      'scope':'TRAIN spatial tiles from ONE specimen; no VAL/TEST accessed by this script',
      'purpose':'Diagnose predicted foreground topology and oracle-majority reference-guided connectivity, not performance ranking',
      'warning':'Reference IDs decide oracle graph merges and are author watershed pseudo-instances, NOT independent truth.',
      'parameters':{'block_zyx':[args.region_z,args.region_xy,args.region_xy],
        'foreground_threshold':args.foreground_threshold,'score_threshold_logit':args.score_threshold,
        'max_negative_fraction':args.max_negative_fraction,'min_faces':args.min_faces,'z_weight':args.z_weight},
      'tiles':results,
      'limitations':['Oracle-majority connected PQ is NOT a mathematical ceiling; supervoxels can straddle reference instances.',
        'Oracle majority can mislabel mixed supervoxels and reference background voxels.',
        'High oracle-connected PQ uses reference labels; cannot be claimed as a model outcome.',
        'Unconstrained majority ID reuse does not guarantee physical connectivity.',
        'TRAIN spatial tiles may overlap fitting crops. Only one specimen.',
        'Z registration and physical voxel spacing unverified.']}
    (args.out/'region_oracle_connectivity.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    pd.DataFrame(table).to_csv(args.out/'region_oracle_edge_filters.csv',index=False)
    print('Saved:',args.out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--split-run-dir',type=Path,required=True)
    parser.add_argument('--affinity-run-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-tiles',type=int,default=1)
    parser.add_argument('--region-z',type=int,default=1)
    parser.add_argument('--region-xy',type=int,default=4)
    parser.add_argument('--z-weight',type=float,default=.25)
    parser.add_argument('--score-threshold',type=float,default=.2)
    parser.add_argument('--max-negative-fraction',type=float,default=.35)
    parser.add_argument('--min-faces',type=int,default=2)
    parser.add_argument('--foreground-threshold',type=float,default=.5)
    parser.add_argument('--guard',type=int,default=12)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--accept-provisional-registration',action='store_true')
    run(parser.parse_args())

if __name__=='__main__':
    main()
