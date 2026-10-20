#!/usr/bin/env python3
"""Focused Z-discontinuity ablation on pre-existing frozen Learned Logistic RAG.

The fixed two rescue strategies are geometry-only mutual-best matching and
geometry with a provisional Z-affinity gate. No hyperparameter sweep, no TEST.
Use TRAIN for pipeline smoke; reused VAL is *development* feedback only.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from train_segmentation import read_data
from train_voxel_affinity import make_network,predict_prob
from evaluate_affinity_region_graph import block_supervoxels
from learned_region_graph import load_region_classifier,learned_interface_evidence
from dynamic_region_merging import merge_dynamic_regions
from evaluate_affinity_graph_pilot import sparse_pq
from instance_error_attribution import overlap_objects,directional_faces
from z_continuity_rescue import propose_z_links,rescue_z_links


def run(args):
    if args.split not in ('train','val') or args.max_tiles<1:
        raise ValueError('Only TRAIN/VAL and positive max-tiles are allowed')
    if args.region_z!=1 or args.region_xy!=4:
        raise ValueError('Frozen full-fit classifier was trained with region 1x4x4')
    if not args.accept_provisional_registration:
        raise ValueError('Use --accept-provisional-registration for provisional Z links')
    audit=json.loads((args.split_run_dir/'data_audit.json').read_text())
    if audit.get('stack_order_check',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise ValueError('Stack order has not passed audit')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {args.out}')
    args.out.mkdir(parents=True,exist_ok=True)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    bundle=load_region_classifier(args.predictor_dir/'logistic_balanced_trained_on_all_train.joblib')
    ck=torch.load(args.affinity_run_dir/'xyz_downweighted.pt',map_location='cpu',weights_only=True)
    if ck.get('z_weight',0)<=0:raise RuntimeError('Z-supervised checkpoint needed')
    model=make_network(int(ck['base']),args.split_run_dir/'unet_3d.pt')
    model.load_state_dict(ck['state_dict'],strict=True)
    model=model.to(device).eval()
    img,ref,_,_=read_data(args.data_dir)
    tiles=pd.read_csv(args.split_run_dir/'tile_splits.csv')
    tiles=tiles[tiles.split==args.split].head(args.max_tiles)
    if tiles.empty:raise RuntimeError(f'No {args.split} tiles')
    results=[];faces=[];diagnostics=[]
    for t in tiles.itertuples(index=False):
        y,x,w=int(t.y),int(t.x),int(t.w);g=args.guard
        if 2*g>=w or y<0 or x<0 or y+w>img.shape[1] or x+w>img.shape[2]:
            raise ValueError('Invalid tile crop')
        with torch.inference_mode():p=predict_prob(model,img[:,y:y+w,x:x+w],device)
        pp=p[:,:,g:w-g,g:w-g]
        intensity=img[:,y+g:y+w-g,x+g:x+w-g]
        reference=ref[:,y+g:y+w-g,x+g:x+w-g]
        sv=block_supervoxels(pp[0]>=args.foreground_threshold,(1,4,4))
        evidence,stats=learned_interface_evidence(sv,pp[1:],intensity,bundle,z_weight=args.z_weight)
        initial,merge_diag=merge_dynamic_regions(sv,evidence,score_threshold=args.score_threshold,
            max_negative_fraction=.35,max_component_voxels=args.max_component_voxels,
            min_faces=args.min_faces,use_repulsion=False,phase_order='joint')
        proposal=propose_z_links(initial,pp[1],min_faces=args.z_min_faces,
            min_overlap=args.min_overlap,min_iou=args.min_iou)
        (args.out/f'z_link_candidates_{y}_{x}.csv').write_text(proposal.to_csv(index=False))
        pred_geom,dgeom=rescue_z_links(initial,proposal,require_affinity=False,
                                      max_component_voxels=args.max_component_voxels)
        pred_gated,dgated=rescue_z_links(initial,proposal,require_affinity=True,
                                       min_mean_pz=args.min_mean_pz,
                                       max_component_voxels=args.max_component_voxels)
        tile={'tile_y':y,'tile_x':x,'split':args.split,'base_merge_stats':merge_diag,
              'interface_scoring_stats':stats,'geometry_rescue':dgeom,'affinity_gated_rescue':dgated}
        diagnostics.append(tile)
        for method,pred in [('learned_logistic_baseline',initial),('z_overlap_mutual',pred_geom),
                            ('z_overlap_mutual_with_pz_gate',pred_gated)]:
            pq=sparse_pq(reference,pred)
            summ,obj,refstats,_=overlap_objects(reference,pred)
            if abs(float(pq['pq'])-float(summ['pq']))>1e-7 or pq['tp']!=summ['tp']:
                raise AssertionError('PQ metric parity failed')
            results.append({'split':args.split,'tile_y':y,'tile_x':x,'method':method,
                'pq':pq['pq'],'tp':pq['tp'],'fp':pq['fp'],'fn':pq['fn'],
                'n_reference':summ['n_reference'],'n_predicted':summ['n_predicted'],
                'fragment_candidate':summ['unmatched_predicted_exclusive_categories'].get('fragment_candidate',0),
                'orphan':summ['n_orphan_objects'],
                'multi_reference_unmatched':summ['unmatched_predicted_exclusive_categories'].get('multi_reference_overlap',0),
                'background_dominated_unmatched':summ['unmatched_predicted_exclusive_categories'].get('background_dominated_overlap',0),
                'other_unmatched_overlap':summ['unmatched_predicted_exclusive_categories'].get('other_unmatched_overlap',0),
                'n_fragmented_references':summ['n_reference_with_2plus_substantial_prediction_components'],
                'reference_extra_component_count_proxy':summ['reference_extra_component_count_proxy']})
            f=directional_faces(reference,pred)
            f['split']=args.split;f['tile_y']=y;f['tile_x']=x;f['method']=method
            faces.append(f)
        print(f'{args.split.upper()} ({y},{x}) '+ ' | '.join(
            f'{r["method"]}: PQ={r["pq"]:.4f} N={r["n_predicted"]}' for r in results[-3:]),flush=True)
        np.savez_compressed(args.out/f'z_link_masks_{y}_{x}.npz',
                            baseline=initial,geometry=pred_geom,geometry_pz=pred_gated)
    pd.DataFrame(results).to_csv(args.out/'z_link_rescue_comparison.csv',index=False)
    pd.concat(faces,ignore_index=True).to_csv(args.out/'z_link_axis_breaks.csv',index=False)
    summary={'scope':f'{args.split.upper()} from ONE specimen; TEST never accessed',
        'original_reference':'Author-generated watershed instances, not independent cell truth',
        'predictor_training':'Full-fit TRAIN classifier; TRAIN run is in-sample smoke, not validation',
        'validation':'VAL tiles repeatedly used for development; not independent test',
        'cross_slice_caution':'Z voxel spacing and registration accuracy NOT verified',
        'design':'Original XY merge scores unchanged; post-hoc candidate links only between adjacent Z layers',
        'settings':{'region_zyx':[1,4,4],'z_min_faces':args.z_min_faces,
                    'min_overlap':args.min_overlap,'min_iou':args.min_iou,
                    'min_mean_pz':args.min_mean_pz,'max_component_voxels':args.max_component_voxels},
        'tile_diagnostics':diagnostics}
    (args.out/'z_link_rescue_diagnostics.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','split-run-dir','affinity-run-dir','predictor-dir','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=['train','val'],default='train')
    p.add_argument('--max-tiles',type=int,default=1)
    p.add_argument('--region-z',type=int,default=1)
    p.add_argument('--region-xy',type=int,default=4)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--foreground-threshold',type=float,default=0.5)
    p.add_argument('--z-weight',type=float,default=0.25)
    p.add_argument('--score-threshold',type=float,default=0.2)
    p.add_argument('--min-faces',type=int,default=2)
    p.add_argument('--max-component-voxels',type=int,default=1200)
    p.add_argument('--z-min-faces',type=int,default=2)
    p.add_argument('--min-overlap',type=float,default=0.35)
    p.add_argument('--min-iou',type=float,default=0.15)
    p.add_argument('--min-mean-pz',type=float,default=0.45)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--accept-provisional-registration',action='store_true')
    args=p.parse_args()
    run(args)
if __name__=='__main__': main()
