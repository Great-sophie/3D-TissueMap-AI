#!/usr/bin/env python3
"""Build TRAIN-only region-interface pseudo-label dataset for 3D-TissueMap AI.

Outputs a model-feature table, a separate reference-only label audit, tile
statistics, and an explicit leakage-safe feature whitelist. The labels are
pseudo-labels from an author watershed and Z registration remains provisional.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from train_segmentation import read_data
from train_voxel_affinity import make_network, predict_prob
from evaluate_affinity_region_graph import block_supervoxels
from region_interface_features import FEATURE_COLUMNS, extract_region_interfaces


def add_tile_metadata(df: pd.DataFrame, *, y: int, x: int, w: int):
    df=df.copy()
    df.insert(0,'tile_id',f'train_y{y}_x{x}_w{w}')
    df.insert(1,'tile_y',y)
    df.insert(2,'tile_x',x)
    df.insert(3,'tile_w',w)
    return df


def summarize(df: pd.DataFrame, y: int, x: int, w: int, n_sv: int):
    positive=(df['pseudo_label']==1)
    negative=(df['pseudo_label']==0)
    eligible=(df['eligible_for_training']==1)
    xy=(df['interface_z_faces']==0)
    return {
        'tile_id':f'train_y{y}_x{x}_w{w}',
        'tile_y':y,'tile_x':x,'tile_w':w,
        'n_supervoxels':int(n_sv),'n_interfaces':int(len(df)),
        'n_reliable_xy':int((df['label_status']=='reliable_xy').sum()),
        'n_provisional_z':int((df['label_status']=='provisional_z').sum()),
        'n_unknown':int((df['pseudo_label']==-1).sum()),
        'n_xy_interfaces':int(xy.sum()),
        'n_z_involved_interfaces':int((~xy).sum()),
        'n_train_eligible':int(eligible.sum()),
        'n_train_same':int((positive & eligible).sum()),
        'n_train_different':int((negative & eligible).sum()),
        'n_train_same_kept_by_old_gate':int((positive & eligible & (df['baseline_static_gate']==1)).sum()),
        'n_train_different_kept_by_old_gate':int((negative & eligible & (df['baseline_static_gate']==1)).sum()),
    }


def run(args):
    if not args.accept_provisional_registration:
        raise RuntimeError('Registration and physical Z spacing unverified; use --accept-provisional-registration')
    if args.max_tiles<0 or args.guard<0 or args.region_z<1 or args.region_xy<1:
        raise ValueError('Invalid max-tiles / guard / region size')
    if args.include_z_pseudo_labels:
        print('WARNING: Z pseudo-labels opted in. Do NOT call them physically verified cell links.',flush=True)
    audit_path=args.split_run_dir/'data_audit.json'
    stack_audit=json.loads(audit_path.read_text())
    if stack_audit.get('stack_order_check',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise RuntimeError('FILENAME_ORDER_MATCH stack audit not verified')
    tile_df=pd.read_csv(args.split_run_dir/'tile_splits.csv')
    required={'y','x','w','split'}
    if not required.issubset(tile_df.columns):
        raise ValueError(f'tile_splits.csv must have {sorted(required)}')
    train=tile_df.loc[tile_df.split=='train'].copy()
    if args.max_tiles:
        train=train.head(args.max_tiles)
    if train.empty:
        raise RuntimeError('No TRAIN tiles in manifest')
    if train[['y','x','w']].duplicated().any():
        raise RuntimeError('Duplicate TRAIN spatial tile coordinates')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')

    ckpath=args.affinity_run_dir/'xyz_downweighted.pt'
    ck=torch.load(ckpath,map_location='cpu',weights_only=True)
    if float(ck.get('z_weight',0))<=0:
        raise RuntimeError('XYZ-supervised checkpoint required (Z metadata collected separately)')
    model=make_network(int(ck['base']),args.split_run_dir/'unet_3d.pt')
    model.load_state_dict(ck['state_dict'],strict=True)
    model=model.to(device).eval()
    img,ref,_,_=read_data(args.data_dir)
    args.out.mkdir(parents=True,exist_ok=True)
    feature_file=args.out/'region_interfaces_train.csv'
    audit_file=args.out/'region_interface_reference_audit.csv'
    labeled_file=args.out/'region_interface_labeled_train.csv'
    for path in (feature_file,audit_file,labeled_file):
        if path.exists():
            raise FileExistsError(f'Output exists, choose a NEW --out directory or move existing file: {path}')
    summaries=[]
    first=True
    for tile in train.itertuples(index=False):
        y,x,w=map(int,(tile.y,tile.x,tile.w))
        g=args.guard
        if w<=2*g or y<0 or x<0 or y+w>img.shape[1] or x+w>img.shape[2]:
            raise ValueError(f'Invalid tile coordinates/guard: {(y,x,w,g)}')
        with torch.inference_mode():
            prob=predict_prob(model,img[:,y:y+w,x:x+w],device)
        if prob.shape != (4,img.shape[0],w,w):
            raise RuntimeError(f'Unexpected model output shape {prob.shape}')
        yc=slice(y+g,y+w-g)
        xc=slice(x+g,x+w-g)
        inner_prob=prob[:,:,g:w-g,g:w-g]
        inner_ref=ref[:,yc,xc]
        inner_img=img[:,yc,xc]
        fg=inner_prob[0]>=args.foreground_threshold
        sv=block_supervoxels(fg,(args.region_z,args.region_xy,args.region_xy))
        feat,audit=extract_region_interfaces(
            sv,inner_prob[1:],inner_img,inner_ref,
            z_weight=args.z_weight,min_coverage=args.min_reference_coverage,
            min_purity=args.min_reference_purity,
            min_overlap_voxels=args.min_reference_overlap_voxels,
            include_z_labels=args.include_z_pseudo_labels,
        )
        summaries.append(summarize(feat,y,x,w,int(sv.max())))
        feat=add_tile_metadata(feat,y=y,x=x,w=w)
        audit=add_tile_metadata(audit,y=y,x=x,w=w)
        feat.to_csv(feature_file,index=False,mode='a',header=first)
        audit.to_csv(audit_file,index=False,mode='a',header=first)
        feat.loc[feat.eligible_for_training==1, ['tile_id',*FEATURE_COLUMNS,'pseudo_label']].to_csv(
            labeled_file,index=False,mode='a',header=first)
        first=False
        s=summaries[-1]
        print(f'TRAIN ({y},{x}): SV {s["n_supervoxels"]}; interfaces {s["n_interfaces"]}; '
              f'eligible XY={s["n_train_eligible"]}, same={s["n_train_same"]}, '
              f'different={s["n_train_different"]}; provisional Z={s["n_provisional_z"]}; '
              f'unknown={s["n_unknown"]}',flush=True)
    pd.DataFrame(summaries).to_csv(args.out/'region_interface_tile_summary.csv',index=False)
    totals={key:int(sum(s[key] for s in summaries)) for key in (
        'n_supervoxels','n_interfaces','n_reliable_xy','n_provisional_z',
        'n_unknown','n_xy_interfaces','n_z_involved_interfaces',
        'n_train_eligible','n_train_same','n_train_different',
        'n_train_same_kept_by_old_gate','n_train_different_kept_by_old_gate')}
    results={
        'scope':'TRAIN manifest tiles only for feature extraction; one specimen. read_data loads full stack into memory; VAL/TEST tiles are NOT selected.',
        'purpose':'Pseudo-labeled spatial region-interface table, NOT a trained merge predictor or generalization result',
        'source_checkpoint':str(ckpath),
        'reference':'Author-produced watershed pseudo instances, NOT independent manual cell ground truth',
        'registration':'Slice-to-mask registration and physical Z spacing unverified',
        'parameters':{
            'block_zyx':[args.region_z,args.region_xy,args.region_xy],
            'foreground_threshold':args.foreground_threshold,'guard':args.guard,
            'z_weight':args.z_weight,'min_reference_coverage':args.min_reference_coverage,
            'min_reference_purity':args.min_reference_purity,
            'min_reference_overlap_voxels':args.min_reference_overlap_voxels,
            'include_z_pseudo_labels':args.include_z_pseudo_labels,
        },
        'label_definition':{
            '1':'Both high-confidence majority reference IDs equal (pseudo same instance)',
            '0':'Both high-confidence majority reference IDs different (pseudo different instance)',
            '-1':'One or both regions have insufficient reference support; UNKNOWN, never automatically negative',
            'provisional_z':'Reference-derived across-slice links; default eligible_for_training=0',
        },
        'classification_feature_whitelist':FEATURE_COLUMNS,
        'never_use_as_predictors':['pseudo_label','label_status','eligible_for_training','baseline_static_gate','sv_a','sv_b',
            'tile_id','tile_y','tile_x','tile_w','reference_*','majority_id_*'],
        'notes':[
            'Labels are derived from author watershed masks; not true independently annotated cells.',
            'Feature CSV contains no reference majority IDs or coverage/purity; reference-only audit is separated.',
            'Training-ready CSV includes only tile_id, the whitelisted predictor columns and pseudo_label; tile_id is for grouped splitting only.',
            'Unlabeled/provisional interfaces remain in the feature CSV for inference and coverage audits.',
            'Do not use random edge-level train/test split. Keep complete spatial tiles as groups.',
            'Affinity network already trained on TRAIN tiles; a second-stage classifier assessed here would be in-sample without new held-out specimens.',
            'XY vs Z links are intrinsically different; Z training disabled by default.',
        ],
        'n_train_tiles':len(summaries),'totals':totals,'tiles':summaries,
    }
    (args.out/'region_interface_dataset_summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print('TOTAL:',json.dumps(totals,ensure_ascii=False),flush=True)
    print('Saved:',args.out,flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--split-run-dir',type=Path,required=True)
    p.add_argument('--affinity-run-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-tiles',type=int,default=0,help='0 = all TRAIN tiles; 2 recommended for first smoke check')
    p.add_argument('--region-z',type=int,default=1)
    p.add_argument('--region-xy',type=int,default=4)
    p.add_argument('--foreground-threshold',type=float,default=0.5)
    p.add_argument('--z-weight',type=float,default=0.25)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--min-reference-coverage',type=float,default=0.70)
    p.add_argument('--min-reference-purity',type=float,default=0.90)
    p.add_argument('--min-reference-overlap-voxels',type=int,default=3)
    p.add_argument('--include-z-pseudo-labels',action='store_true',
                   help='OPTIONAL exploratory cross-Z training eligibility; default OFF due provisional registration')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--accept-provisional-registration',action='store_true')
    run(p.parse_args())

if __name__=='__main__':
    main()
