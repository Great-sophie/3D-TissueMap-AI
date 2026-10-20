#!/usr/bin/env python3
"""TRAIN-only fixed-patch overfitting diagnostic for 3D voxel affinity learning.

Requires existing scripts/train_segmentation.py and scripts/train_voxel_affinity.py.
Selects exactly ONE deterministic patch from the TRAIN tile manifest and
repeats updates on that patch. No validation/test data are read or scored.
The run diagnoses whether an affinity network can fit its pseudo labels;
it does NOT estimate transfer to new tissue or instance reconstruction PQ.
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

from train_voxel_affinity import (
    AXES, affinity_targets, conditional_affinity_loss,
    edge_stats, make_network, predict_prob, read_data
)


def select_training_patch(img, lab, train_tiles, patch_z, patch_xy, guard, seed, trials=180):
    """Select a boundary-containing patch using ONLY TRAIN tiles."""
    if not train_tiles: raise ValueError('No train tiles in manifest')
    rng=np.random.default_rng(seed)
    choice=None
    best_score=-np.inf
    for _ in range(trials):
        t=train_tiles[int(rng.integers(0, len(train_tiles)))]
        y0,x0,w=int(t['y']),int(t['x']),int(t['w'])
        if patch_xy>w-2*guard:raise ValueError('patch_xy too large for guarded tile')
        z=int(rng.integers(0,img.shape[0]-patch_z+1))
        y=int(rng.integers(y0+guard,y0+w-guard-patch_xy+1))
        x=int(rng.integers(x0+guard,x0+w-guard-patch_xy+1))
        masks=lab[z:z+patch_z,y:y+patch_xy,x:x+patch_xy]
        frac=float((masks>0).mean())
        if frac<0.035 or frac>0.8:
            continue
        targets,valid=affinity_targets(masks)
        diff=(valid & (targets<0.5)).sum(axis=(1,2,3)).astype(int)
        same=(valid & (targets>0.5)).sum(axis=(1,2,3)).astype(int)
        # Keep all three supervised directions present when feasible.
        have_all=bool((diff>=8).all() and (same>=8).all())
        score=(100000 if have_all else 0)+min(int(diff[1]),int(diff[2]))*10+int(diff[0])*0.3 + frac*10
        if score>best_score:
            best_score=score
            choice=(img[z:z+patch_z,y:y+patch_xy,x:x+patch_xy].copy(),
                    masks.copy(), {'tile_y':y0,'tile_x':x0,'tile_width':w,
                                   'z':z,'y':y,'x':x,'reference_foreground_fraction':frac,
                                   'same_edges':same.tolist(),'different_edges':diff.tolist(),
                                   'all_axes_have_both_classes':have_all})
    if choice is None:
        raise RuntimeError('No tissue-containing TRAIN patch found; increase trials or change seed')
    if not choice[2]['all_axes_have_both_classes']:
        raise RuntimeError('Could not find a TRAIN patch with >=8 same and >=8 different adjacency pairs in each axis; increase --trials or change --seed')
    return choice


def metrics_from_model(model,img,label,device):
    output=predict_prob(model,img,device)
    rows=edge_stats(output[1:],label)
    return output, {r['axis']:r for r in rows}


def dump_figures(output_dir,img,labels,p0,p1):
    same,valid=affinity_targets(labels)
    z=img.shape[0]//2
    fig,ax=plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
    ax[0].imshow(img[z],cmap='gray',vmin=0,vmax=1)
    ax[0].set_title('Ir193 (one TRAIN Z plane)')
    ax[1].imshow(labels[z],cmap='nipy_spectral',interpolation='nearest')
    ax[1].set_title('Watershed pseudo-instance IDs')
    for a in ax:a.axis('off')
    fig.savefig(output_dir/'fixed_patch_input.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,10),constrained_layout=True)
    for i,name in enumerate(AXES):
        gt=np.where(valid[i,z],1-same[i,z],np.nan)
        before=np.where(valid[i,z],p0[i+1,z],np.nan)
        after=np.where(valid[i,z],p1[i+1,z],np.nan)
        for j,(pic,title) in enumerate(((gt,'Pseudo different-ID edge'),(before,'Before: P(same)'),(after,'After: P(same)'))):
            axes[i,j].imshow(pic,cmap='viridis',vmin=0,vmax=1,interpolation='nearest')
            axes[i,j].set_title(f'{name}: {title}',fontsize=11)
            axes[i,j].axis('off')
    fig.suptitle('TRAIN patch ONLY | masked foreground-foreground edges | pseudo labels',fontsize=12)
    fig.savefig(output_dir/'affinity_fixed_patch_before_after.png',dpi=150)
    plt.close(fig)


def run(args):
    if args.steps<1 or args.patch_z<4 or args.patch_z%2 or args.patch_xy<16 or args.patch_xy%2:
        raise ValueError('Require steps >=1 and even patch_z >=4, patch_xy >=16')
    if args.z_weight<=0:raise ValueError('For 3-axis diagnostic require z_weight >0')
    if args.log_every<1 or args.trials<1:raise ValueError('log_every and trials must be positive')
    if args.device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    else:device=args.device
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    audit=json.loads((args.run_dir/'data_audit.json').read_text())
    if audit.get('stack_order_check',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise RuntimeError('Stack order not filename-verified; stop')
    if not args.accept_provisional_registration:
        raise RuntimeError('Acknowledge provisional image-to-label registration using --accept-provisional-registration')
    splits=pd.read_csv(args.run_dir/'tile_splits.csv')
    if not {'y','x','w','split'}.issubset(splits.columns):raise ValueError('Malformed split manifest')
    tiles=splits[splits['split']=='train'].to_dict('records')
    args.out.mkdir(parents=True,exist_ok=True)
    img,labels,_,_=read_data(args.data_dir)
    x_np,y_np,location=select_training_patch(img,labels,tiles,args.patch_z,args.patch_xy,args.guard,args.seed,args.trials)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model=make_network(args.base,args.run_dir/'unet_3d.pt').to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr)
    p_before, m_before=metrics_from_model(model,x_np,y_np,device)
    x=torch.from_numpy(x_np.astype(np.float32).copy())[None,None].to(device)
    history=[]
    for step in range(1,args.steps+1):
        model.train()
        optim.zero_grad(set_to_none=True)
        loss,details=conditional_affinity_loss(model(x),y_np,args.z_weight)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward()
        optim.step()
        if step==1 or step%args.log_every==0 or step==args.steps:
            _,metrics=metrics_from_model(model,x_np,y_np,device)
            line={'step':step,'total_loss':float(loss.detach()),
                  'semantic_loss':details['semantic'],'affinity_loss':details['affinity']}
            for ax in AXES:
                r=metrics[ax]
                line[f'{ax}_balanced_accuracy']=r['balanced_accuracy']
                line[f'{ax}_different_recall']=r['different_recall']
                line[f'{ax}_mean_prob_same']=r.get('mean_same_prob_on_true_same')
                line[f'{ax}_mean_prob_different']=r.get('mean_same_prob_on_true_different')
            history.append(line)
            summary=' '.join(f'{a}_BA={metrics[a]["balanced_accuracy"]:.3f}' for a in AXES)
            print(f'step {step:4d}/{args.steps}: loss={float(loss.detach()):.4f} aff={details["affinity"]:.4f} {summary}',flush=True)
    p_after,m_after=metrics_from_model(model,x_np,y_np,device)
    pd.DataFrame(history).to_csv(args.out/'fixed_patch_history.csv',index=False)
    dump_figures(args.out,x_np,y_np,p_before,p_after)
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'base':args.base,'training_patch':location,'steps':args.steps,
                'scope':'TRAIN ONE PATCH ONLY - DO NOT REPORT AS VALIDATION'},args.out/'affinity_fixed_patch_diagnostic.pt')
    result={
        'purpose':'Fixed one TRAIN patch overfitting / optimizer diagnostic',
        'scope':'TRAIN only; no VAL or TEST tiles used for fitting or scoring',
        'data':'Author watershed pseudo instances from one 3D IMC specimen',
        'interpretation':'Cannot establish patient generalization, independent truth, instance PQ or verified physical cell contact',
        'patch':location,'patch_shape_ZYX':list(y_np.shape),
        'steps':args.steps,'z_weight':args.z_weight,'seed':args.seed,'learning_rate':args.lr,
        'before':m_before,'after':m_after,
        'method_note':'P(same instance) at threshold 0.5, conditioned on both neighbor voxels being pseudo-foreground',
        'cautions':['Image-mask registration remains provisional',
                    'Voxel spacing unverified; Z edges may have noisier pseudo labels',
                    'Fit on ONE selected patch can be memorization, not generalization',
                    'For Z edge visualizations, displayed plane is lower endpoint']}
    (args.out/'affinity_fixed_patch_check.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('\n=== TRAIN PATCH BEFORE / AFTER ===')
    for ax in AXES:
        b=m_before[ax];a=m_after[ax]
        print(f'{ax}: balanced_accuracy {b["balanced_accuracy"]:.4f} -> {a["balanced_accuracy"]:.4f}; '
              f'different_recall {b["different_recall"]:.4f} -> {a["different_recall"]:.4f}; '
              f'P(same)|same {b["mean_same_prob_on_true_same"]:.3f} -> {a["mean_same_prob_on_true_same"]:.3f}; '
              f'P(same)|different {b["mean_same_prob_on_true_different"]:.3f} -> {a["mean_same_prob_on_true_different"]:.3f}')
    print('Saved:',args.out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--patch-z',type=int,default=8)
    p.add_argument('--patch-xy',type=int,default=64)
    p.add_argument('--base',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--trials',type=int,default=180)
    p.add_argument('--z-weight',type=float,default=.25)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--log-every',type=int,default=25)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    p.add_argument('--accept-provisional-registration',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
