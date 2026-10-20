#!/usr/bin/env python3
"""Exploratory 3D instance PQ with learned XY region edges and raw Z fallback.

Run TRAIN for smoke testing; report existing VAL separately as reused development,
NEVER unseen holdout. The source pseudo-labels are author-generated watershed.
No reference labels or pseudo-label audits are accessed by graph inference.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate_affinity_region_graph import block_supervoxels, interface_edges, merge_regions
from dynamic_region_merging import build_region_evidence, merge_dynamic_regions
from learned_region_graph import load_region_classifier, learned_interface_evidence
from train_segmentation import read_data
from train_voxel_affinity import make_network, predict_prob
from instance_watershed_baseline import segment
from evaluate_affinity_graph_pilot import sparse_pq, component_stats


def run(args):
    if args.split not in ('train','val') or args.max_tiles<1:
        raise ValueError('Only positive tile count and TRAIN/VAL are supported; never TEST')
    if args.region_z!=1 or args.region_xy!=4:
        raise ValueError('Current full-fit predictor was trained on 1x4x4 supervoxels only')
    if not args.accept_provisional_registration:
        raise RuntimeError('Use --accept-provisional-registration to acknowledge unverified image alignment')
    audit=json.loads((args.split_run_dir/'data_audit.json').read_text())
    if audit.get('stack_order_check',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise RuntimeError('Stack-order audit not passed')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {args.out}')
    args.out.mkdir(parents=True,exist_ok=True)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    models={}
    for name in ('logistic_balanced','hgb_balanced'):
        models[name]=load_region_classifier(args.predictor_dir/f'{name}_trained_on_all_train.joblib')
    img,ref,_,_=read_data(args.data_dir)
    tiles=pd.read_csv(args.split_run_dir/'tile_splits.csv')
    tiles=tiles[tiles.split==args.split].head(args.max_tiles)
    if tiles.empty:
        raise RuntimeError(f'No {args.split.upper()} tiles')
    ck=torch.load(args.affinity_run_dir/'xyz_downweighted.pt',map_location='cpu',weights_only=True)
    if ck.get('z_weight',0)<=0:
        raise RuntimeError('XYZ-supervised checkpoint required for provisional Z fallback')
    model=make_network(int(ck['base']),args.split_run_dir/'unet_3d.pt')
    model.load_state_dict(ck['state_dict'],strict=True)
    model=model.to(device).eval()
    rows=[]; tile_diagnostics=[]
    for t in tiles.itertuples(index=False):
        y,x,w=int(t.y),int(t.x),int(t.w)
        g=args.guard
        if 2*g>=w or x<0 or y<0 or y+w>img.shape[1] or x+w>img.shape[2]:
            raise ValueError('Invalid tile/guard')
        with torch.inference_mode():
            p=predict_prob(model,img[:,y:y+w,x:x+w],device)
        crop=p[:,:,g:w-g,g:w-g]
        image=img[:,y+g:y+w-g,x+g:x+w-g]
        reference=ref[:,y+g:y+w-g,x+g:x+w-g]
        sv=block_supervoxels(crop[0]>=args.foreground_threshold,(args.region_z,args.region_xy,args.region_xy))
        orig_edges=interface_edges(sv,crop[1:],args.z_weight)
        original_evidence=build_region_evidence(sv,crop[1:],args.z_weight)
        if len(orig_edges)!=len(original_evidence):
            raise AssertionError('Mismatch in original edge definitions')
        common={'score_threshold':args.score_threshold,
                'max_negative_fraction':args.max_negative_fraction,
                'min_faces':args.min_faces,
                'max_component_voxels':args.max_component_voxels}
        predictions=[]; diag={'split':args.split,'tile_y':y,'tile_x':x,
                              'n_initial_sv':int(sv.max()),'n_region_edges':int(len(orig_edges))}
        ws=segment(crop[0],mode='watershed',threshold=args.foreground_threshold,min_distance=3)
        predictions.append(('distance_watershed',ws,None))
        static_pred,nstatic=merge_regions(sv,orig_edges,args.score_threshold,args.max_negative_fraction,
                                          args.max_component_voxels,args.min_faces)
        predictions.append(('static_greedy',static_pred,{'n_merges':nstatic}))
        dy_pred,dy_diag=merge_dynamic_regions(sv,original_evidence,**common,
                                             use_repulsion=True,phase_order='joint')
        predictions.append(('dynamic_original_repulsion',dy_pred,dy_diag))
        for name,bundle in models.items():
            learned,scoring_stats=learned_interface_evidence(sv,crop[1:],image,bundle,z_weight=args.z_weight)
            diag[name+'_scores']=scoring_stats
            for veto in (False,True):
                method=f'{name}_frozen_edge_'+('raw_repulsion' if veto else 'score_only')
                pred,info=merge_dynamic_regions(sv,learned,**common,
                                                use_repulsion=veto,phase_order='joint')
                predictions.append((method,pred,info))
                diag[method]=info
        for method,labels,info in predictions:
            rows.append({'split':args.split,'tile_y':y,'tile_x':x,'method':method,
                         **sparse_pq(reference,labels),**component_stats(labels),
                         'n_initial_sv':int(sv.max()),
                         'n_edges':int(len(orig_edges)),
                         'n_merges':None if info is None else info['n_merges']})
        tile_diagnostics.append(diag)
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        z=int(crop.shape[1]//2)
        axes.flat[0].imshow(image[z],cmap='gray')
        axes.flat[0].set_title('Ir193 input',fontsize=9)
        for ax,(name,labels,_) in zip(list(axes.flat)[1:],predictions):
            ax.imshow(np.ma.masked_equal(labels[z],0),cmap='nipy_spectral',interpolation='nearest')
            score=next(r['pq'] for r in rows if r['tile_y']==y and r['tile_x']==x and r['method']==name)
            ax.set_title(f'{name}\nPQ={score:.4f}',fontsize=7)
        for ax in axes.flat: ax.axis('off')
        fig.tight_layout()
        fig.savefig(args.out/f'learned_region_comparison_{y}_{x}.png',dpi=120)
        plt.close(fig)
        print(f'{args.split.upper()} ({y},{x}) '+ ' | '.join(
            f'{method} PQ={next(r["pq"] for r in rows if r["tile_y"]==y and r["tile_x"]==x and r["method"]==method):.4f} N={int((labels>0).any()) if False else len(np.unique(labels[labels>0]))}'
            for method,labels,_ in predictions),flush=True)
    pd.DataFrame(rows).to_csv(args.out/'learned_region_instance_comparison.csv',index=False)
    summary={'scope':f'{args.split.upper()} spatial tiles from ONE specimen; TEST never selected',
             'warning':'Author-watershed pseudo instances; no manual truth or verified physical Z spacing',
             'predictor_dir':str(args.predictor_dir),
             'predictor_training':'Both loaded classifiers were fit on ALL TRAIN tiles',
             'in_sample_warning':bool(args.split=='train'),
             'reused_validation_warning':bool(args.split=='val'),
             'reference_used_by_graph_inference':False,
             'xy_classifier':'All pure XY interfaces, including previously unlabeled/unknown interfaces',
             'z_interface_fallback':'Original XYZ voxel-affinity evidence; no XY classifier inference on any Z-involved interface',
             'frozen_edge_warning':'Atomic learned interface scores are weighted-aggregated after merges; classifier is NOT rerun on enlarged regions',
             'uncalibrated_warning':'Balanced-class training probabilities are not verified as calibrated; threshold 0.2 is provisional, not optimized',
             'rows':rows,'diagnostics':tile_diagnostics}
    (args.out/'learned_region_instance_comparison.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved:',args.out)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--split-run-dir',type=Path,required=True)
    p.add_argument('--affinity-run-dir',type=Path,required=True)
    p.add_argument('--predictor-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['train','val'],default='train')
    p.add_argument('--max-tiles',type=int,default=1)
    p.add_argument('--region-z',type=int,default=1)
    p.add_argument('--region-xy',type=int,default=4)
    p.add_argument('--z-weight',type=float,default=0.25)
    p.add_argument('--score-threshold',type=float,default=0.2)
    p.add_argument('--max-negative-fraction',type=float,default=0.35)
    p.add_argument('--max-component-voxels',type=int,default=1200)
    p.add_argument('--min-faces',type=int,default=2)
    p.add_argument('--foreground-threshold',type=float,default=0.5)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--accept-provisional-registration',action='store_true')
    args=p.parse_args()
    if args.min_faces<1 or args.max_component_voxels<1:
        p.error('min-faces and max-component-voxels must be positive')
    run(args)

if __name__=='__main__':main()
