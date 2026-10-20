#!/usr/bin/env python3
"""Frozen-parameter diagnostic of dynamic region merging versus static greedy.

Only user-chosen TRAIN/VAL spatial tiles, never TEST. The specimen and pseudo
watershed labels are NOT independent truth. No data from reference labels
enter graph construction or merge decisions.
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
from train_segmentation import read_data
from train_voxel_affinity import make_network, predict_prob
from instance_watershed_baseline import segment
from evaluate_affinity_graph_pilot import sparse_pq, component_stats


def run(args):
    if args.max_tiles < 1:
        raise ValueError('--max-tiles must be >=1')
    if not args.accept_provisional_registration:
        raise RuntimeError('Use --accept-provisional-registration to acknowledge unverified image registration')
    audit=json.loads((args.split_run_dir/'data_audit.json').read_text())
    if audit.get('stack_order_check',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise RuntimeError('Stack-order audit not passed')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    args.out.mkdir(parents=True,exist_ok=True)
    img,ref,_,_=read_data(args.data_dir)
    tiles=pd.read_csv(args.split_run_dir/'tile_splits.csv')
    tiles=tiles[tiles.split==args.split].head(args.max_tiles)
    if tiles.empty:
        raise RuntimeError(f'No spatial tiles for {args.split}')
    ck=torch.load(args.affinity_run_dir/'xyz_downweighted.pt',map_location='cpu',weights_only=True)
    if ck.get('z_weight',0)<=0:
        raise RuntimeError('Use an XYZ-supervised model')
    model=make_network(int(ck['base']),args.split_run_dir/'unet_3d.pt')
    model.load_state_dict(ck['state_dict'],strict=True)
    model=model.to(device).eval()
    allrows=[]
    diagnostics=[]
    for t in tiles.itertuples(index=False):
        y,x,w=int(t.y),int(t.x),int(t.w)
        g=args.guard
        if 2*g >= w:
            raise ValueError('Guard too large for tile')
        with torch.inference_mode():
            p=predict_prob(model,img[:,y:y+w,x:x+w],device)
        p=p[:,:,g:w-g,g:w-g]
        reference=ref[:,y+g:y+w-g,x+g:x+w-g]
        sv=block_supervoxels(p[0]>=args.foreground_threshold, (args.region_z,args.region_xy,args.region_xy))
        edges=interface_edges(sv,p[1:],args.z_weight)
        evidence=build_region_evidence(sv,p[1:],args.z_weight)
        if len(edges)!=len(evidence):
            raise AssertionError('Static and dynamic graphs must have same number of original interfaces')
        static_pred, nstatic=merge_regions(sv,edges,args.score_threshold,args.max_negative_fraction,
                                          args.max_component_voxels,args.min_faces)
        common=dict(score_threshold=args.score_threshold,
                    max_negative_fraction=args.max_negative_fraction,
                    max_component_voxels=args.max_component_voxels,min_faces=args.min_faces)
        dynamic_plain,plain_info=merge_dynamic_regions(
            sv,evidence,**common,use_repulsion=False,phase_order='joint')
        dynamic_joint,joint_info=merge_dynamic_regions(
            sv,evidence,**common,use_repulsion=True,phase_order='joint')
        dynamic_staged,staged_info=merge_dynamic_regions(
            sv,evidence,**common,use_repulsion=True,phase_order='xy_then_z')
        ws=segment(p[0],mode='watershed',threshold=args.foreground_threshold,min_distance=3)
        images=[('distance_watershed',ws),('static_greedy',static_pred),
                ('dynamic_joint_score_only',dynamic_plain),
                ('dynamic_joint_repulsion',dynamic_joint),
                ('dynamic_xy_then_z_repulsion',dynamic_staged)]
        for name,lab in images:
            row={'split':args.split,'tile_y':y,'tile_x':x,'method':name,
                 **sparse_pq(reference,lab),**component_stats(lab),
                 'n_supervoxels':int(sv.max()),'n_interfaces':int(len(edges)),
                 'n_merges': {'distance_watershed':None,'static_greedy':nstatic,
                              'dynamic_joint_score_only':plain_info['n_merges'],
                              'dynamic_joint_repulsion':joint_info['n_merges'],
                              'dynamic_xy_then_z_repulsion':staged_info['n_merges']}[name]}
            allrows.append(row)
        diagnostics.append({'tile_y':y,'tile_x':x,'split':args.split,
                            'static_merges':int(nstatic),
                            'dynamic_joint_score_only':plain_info,
                            'dynamic_joint_repulsion':joint_info,
                            'dynamic_xy_then_z_repulsion':staged_info})
        fig, axs=plt.subplots(1,6,figsize=(21,4))
        z=int(p.shape[1]//2)
        axs[0].imshow(img[z,y+g:y+w-g,x+g:x+w-g],cmap='gray')
        axs[0].set_title('Input Ir193')
        for ax,(name,lab) in zip(axs[1:],images):
            ax.imshow(np.ma.masked_equal(lab[z],0),cmap='nipy_spectral',interpolation='nearest')
            ax.set_title(f'{name}\nPQ {next(r["pq"] for r in allrows[::-1] if r["method"]==name):.3f}',fontsize=8)
        for ax in axs:ax.axis('off')
        fig.suptitle(f'{args.split.upper()} / one specimen / watershed pseudo reference / dynamic RAG diagnostic',fontsize=11)
        fig.tight_layout()
        fig.savefig(args.out/f'dynamic_region_comparison_{y}_{x}.png',dpi=140)
        plt.close(fig)
        print(f'{args.split.upper()} ({y},{x}) SV={sv.max()} edges={len(edges)}: '+
              ', '.join(f'{r["method"]} PQ={r["pq"]:.4f} N={r["n_predicted"]}' for r in allrows[-5:]),flush=True)
    pd.DataFrame(allrows).to_csv(args.out/'dynamic_region_validation.csv',index=False)
    result={'scope':f'{args.split.upper()} spatial tiles, ONE specimen; TEST never accessed',
            'reference':'Author watershed pseudo-instances, not independent manual annotation',
            'methods': ['Static original greedy with repulsion veto',
                        'Dynamic joint score only',
                        'Dynamic joint with repulsion veto',
                        'Dynamic XY-first / joint-second with repulsion veto'],
            'model_checkpoint':str(args.affinity_run_dir/'xyz_downweighted.pt'),
            'params': {'block_zyx':[args.region_z,args.region_xy,args.region_xy],
                       'phase_order_options':['joint','xy_then_z'],'z_weight':args.z_weight,
                       'score_threshold_logit':args.score_threshold,
                       'max_negative_fraction':args.max_negative_fraction,
                       'min_faces':args.min_faces,
                       'max_component_voxels':args.max_component_voxels,
                       'foreground_threshold':args.foreground_threshold},
            'rows':allrows,'diagnostics':diagnostics,
            'cautions':['Static and dynamic region methods share identical prediction and foreground.',
                        'Repulsive veto is an interface-level heuristic; NOT global mutex or must-not-link.',
                        'Dynamic pooled score may become LOWER after merges and worsen fragmentation.',
                        'PQ is against author-produced watershed pseudo labels.',
                        'All tiles from one specimen, possible spatial correlation.',
                        'Previously inspected VAL is not an independent final holdout.',
                        'Physical voxel spacing and stack registration not verified.']}
    (args.out/'dynamic_region_comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('Saved:',args.out,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--split-run-dir',type=Path,required=True)
    ap.add_argument('--affinity-run-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--split',choices=['train','val'],default='train')
    ap.add_argument('--max-tiles',type=int,default=1)
    ap.add_argument('--region-z',type=int,default=1)
    ap.add_argument('--region-xy',type=int,default=4)
    ap.add_argument('--z-weight',type=float,default=0.25)
    ap.add_argument('--score-threshold',type=float,default=0.2)
    ap.add_argument('--max-negative-fraction',type=float,default=0.35)
    ap.add_argument('--max-component-voxels',type=int,default=1200)
    ap.add_argument('--min-faces',type=int,default=2)
    ap.add_argument('--foreground-threshold',type=float,default=0.5)
    ap.add_argument('--guard',type=int,default=12)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--accept-provisional-registration',action='store_true')
    run(ap.parse_args())

if __name__=='__main__':main()
