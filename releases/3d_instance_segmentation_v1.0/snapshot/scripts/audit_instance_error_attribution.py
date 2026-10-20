#!/usr/bin/env python3
"""Audit missed 3D cell instance matches, fragments, mixed-ID objects, XY/Z face breaks.

Existing learned Logistic + old watershed on TRAIN (in-sample smoke) or already
reused VAL (development diagnostic only). No TEST support. No reference used at
inference. Same exact network, crop, checkpoint & learned decoder as previous.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from instance_error_attribution import overlap_objects,directional_faces
from evaluate_affinity_region_graph import block_supervoxels,interface_edges,merge_regions
from dynamic_region_merging import merge_dynamic_regions
from learned_region_graph import load_region_classifier,learned_interface_evidence
from train_segmentation import read_data
from train_voxel_affinity import make_network,predict_prob
from instance_watershed_baseline import segment
from evaluate_affinity_graph_pilot import sparse_pq


def run(args):
    if args.split not in ('train','val') or args.max_tiles<1:
        raise ValueError('Only TRAIN or reused VAL, positive max-tiles; TEST forbidden')
    if not args.accept_provisional_registration:raise RuntimeError('Acknowledge provisional registration')
    if args.region_z!=1 or args.region_xy!=4:
        raise ValueError('Full-fit classifier is trained with 1x4x4 region supervoxels')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {args.out}')
    audit=json.loads((args.split_run_dir/'data_audit.json').read_text())
    if audit.get('stack_order_check',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise RuntimeError('Stack-order audit did not pass')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    clf=load_region_classifier(args.predictor_dir/'logistic_balanced_trained_on_all_train.joblib')
    img,ref,_,_=read_data(args.data_dir)
    tiles=pd.read_csv(args.split_run_dir/'tile_splits.csv')
    tiles=tiles[tiles['split']==args.split].head(args.max_tiles)
    if tiles.empty:raise ValueError('No selected TRAIN/VAL tiles')
    ck=torch.load(args.affinity_run_dir/'xyz_downweighted.pt',map_location='cpu',weights_only=True)
    if ck.get('z_weight',0)<=0:raise ValueError('XYZ-supervised affinity checkpoint required')
    model=make_network(int(ck['base']),args.split_run_dir/'unet_3d.pt')
    model.load_state_dict(ck['state_dict'],strict=True)
    model=model.to(device).eval()
    args.out.mkdir(parents=True)
    summary_rows=[]; object_rows=[];reference_rows=[];direction_rows=[]; summary_tiles=[]; category_rows=[]
    for tile in tiles.itertuples(index=False):
        y,x,w=int(tile.y),int(tile.x),int(tile.w);g=args.guard
        if w<=2*g or y<0 or x<0 or y+w>img.shape[1] or x+w>img.shape[2]:
            raise ValueError('Invalid tile or crop')
        with torch.inference_mode():
            probabilities=predict_prob(model,img[:,y:y+w,x:x+w],device)
        crop=probabilities[:,:,g:w-g,g:w-g]
        image=img[:,y+g:y+w-g,x+g:w-g+x]
        reference=ref[:,y+g:y+w-g,x+g:x+w-g]
        sv=block_supervoxels(crop[0]>=args.foreground_threshold,(1,4,4))
        learned,learned_stats=learned_interface_evidence(sv,crop[1:],image,clf,z_weight=args.z_weight)
        score_pred,score_diag=merge_dynamic_regions(sv,learned,score_threshold=args.score_threshold,
                               max_negative_fraction=args.max_negative_fraction,min_faces=args.min_faces,
                               max_component_voxels=args.max_component_voxels,use_repulsion=False,phase_order='joint')
        ws=segment(crop[0],mode='watershed',threshold=args.foreground_threshold,min_distance=3)
        for method,pred in (('distance_watershed',ws),('logistic_balanced_frozen_edge_score_only',score_pred)):
            metrics,objects,refs,overlap=overlap_objects(reference,pred,min_overlap_voxels=args.min_overlap_voxels,
                                                        min_reference_fraction=args.min_reference_fraction)
            legacy=sparse_pq(reference,pred)
            if metrics['tp']!=legacy['tp'] or abs(metrics['pq']-legacy['pq'])>1e-8:
                raise AssertionError(f'PQ mismatch for {method}: {metrics} vs {legacy}')
            index={'split':args.split,'tile_y':y,'tile_x':x,'method':method}
            summary_rows.append({**index,**{k:v for k,v in metrics.items() if not isinstance(v,dict)}})
            for category,n in metrics['unmatched_predicted_exclusive_categories'].items():
                category_rows.append({**index,'category':category,'n_objects':n})
            for item in (objects,refs):
                for k,v in index.items():item[k]=v
            object_rows.append(objects);reference_rows.append(refs)
            faces=directional_faces(reference,pred)
            for k,v in index.items():faces[k]=v
            direction_rows.append(faces)
            summary_tiles.append({**index,**metrics,'face_connectivity':faces.to_dict(orient='records')})
            print(f'{args.split.upper()} ({y},{x}) {method}: N_pred={metrics["n_predicted"]} '
                  f'PQ={metrics["pq"]:.4f} unmatched={metrics["fp"]} '
                  f'orphan={metrics["n_orphan_objects"]} '
                  f'fragmented_reference={metrics["n_reference_with_2plus_substantial_prediction_components"]}',flush=True)
    pd.DataFrame(summary_rows).to_csv(args.out/'instance_error_summary.csv',index=False)
    pd.concat(object_rows,ignore_index=True).to_csv(args.out/'predicted_object_categories.csv',index=False)
    pd.DataFrame(category_rows,columns=['split','tile_y','tile_x','method','category','n_objects']).to_csv(args.out/'unmatched_category_counts.csv',index=False)
    pd.concat(reference_rows,ignore_index=True).to_csv(args.out/'reference_fragmentation.csv',index=False)
    pd.concat(direction_rows,ignore_index=True).to_csv(args.out/'xy_z_face_continuity.csv',index=False)
    result={'scope':f'{args.split.upper()} spatial tiles from ONE specimen; TEST never accessed',
            'predictor':'FULL TRAIN refit Logistic; TRAIN audit is in-sample, VAL already reused for iterative development',
            'reference':'Author-produced watershed pseudo-instances, NOT independent manual ground truth',
            'note':'Unmatched categories are exclusive observational heuristics, not proven causal error types. FP means unmatched at IoU>=0.5, NOT necessarily pure-background.',
            'face_continuity':'Rate of identical prediction ID on same-reference-ID 6-neighbor voxel faces, CONDITIONAL on both voxels predicted foreground; no physical Z verified',
            'inference_uses_reference':False, 'classifier_for_z':'Raw XYZ affinity fallback',
            'model':'runs/voxel_affinity_extended/xyz_downweighted.pt',
            'params':{'region_zyx':[1,4,4],'foreground_threshold':args.foreground_threshold,
                      'score_threshold_logit':args.score_threshold,'min_faces':args.min_faces,
                      'max_component_voxels':args.max_component_voxels,
                      'min_overlap_voxels':args.min_overlap_voxels,
                      'min_reference_fraction':args.min_reference_fraction},
            'tiles':summary_tiles}
    (args.out/'instance_error_attribution.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--split-run-dir',type=Path,required=True)
    p.add_argument('--affinity-run-dir',type=Path,required=True)
    p.add_argument('--predictor-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['train','val'],default='val')
    p.add_argument('--max-tiles',type=int,default=1)
    p.add_argument('--region-z',type=int,default=1)
    p.add_argument('--region-xy',type=int,default=4)
    p.add_argument('--foreground-threshold',type=float,default=.5)
    p.add_argument('--score-threshold',type=float,default=.2)
    p.add_argument('--max-negative-fraction',type=float,default=.35)
    p.add_argument('--min-faces',type=int,default=2)
    p.add_argument('--max-component-voxels',type=int,default=1200)
    p.add_argument('--z-weight',type=float,default=.25)
    p.add_argument('--min-overlap-voxels',type=int,default=5)
    p.add_argument('--min-reference-fraction',type=float,default=.05)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--accept-provisional-registration',action='store_true')
    args=p.parse_args()
    run(args)
if __name__=='__main__':main()
