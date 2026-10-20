#!/usr/bin/env python3
"""Exploratory supervised 3D nearest-neighbor voxel affinity learning.

Runs XY-supervised and XYZ-supervised ablations from the SAME prior 3D U-Net.
TRAIN tiles for optimization; VAL tiles for diagnostic edge classification only.
Does NOT use TEST, infer biological contact, verify registration, or produce PQ.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_segmentation import UNet, read_data

AXES = ('Z','Y','X')


def affinity_targets(labels: np.ndarray):
    """Return [3,Z,Y,X] binary SAME-instance targets and valid foreground pair masks.

    For each axis, edge lives at the lower-indexed endpoint. Last slice/row/column
    in each axis is always invalid, to prevent false wrap-around supervision.
    """
    label=np.asarray(labels)
    if label.ndim!=3 or min(label.shape)<2 or not np.issubdtype(label.dtype,np.integer):
        raise ValueError('Expected integer 3D instance labels with all dimensions >= 2')
    same=np.zeros((3,*label.shape),np.float32)
    valid=np.zeros_like(same,dtype=bool)
    for axis in range(3):
        lo=[slice(None)]*3;hi=[slice(None)]*3
        lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        a=label[tuple(lo)];b=label[tuple(hi)]
        both=(a>0)&(b>0)
        valid[(axis,*lo)]=both
        same[(axis,*lo)]=((a==b)&both).astype(np.float32)
    return same,valid


def conditional_affinity_loss(logits:torch.Tensor, labels:np.ndarray, z_weight:float):
    """Balanced conditional edge BCE; no background involved, no test labels.

    logits: [1,4,Z,Y,X], channel 0 = foreground, 1..3 = SAME affinity.
    Two classes per direction are averaged separately to handle imbalance.
    """
    if logits.ndim!=5 or logits.shape[0]!=1 or logits.shape[1]!=4:
        raise ValueError('Expected batch=1 and four output channels')
    same,valid=affinity_targets(labels)
    if tuple(logits.shape[2:])!=tuple(labels.shape):raise ValueError('Shape mismatch')
    fg=torch.from_numpy((labels>0).astype(np.float32).copy())[None,None].to(logits.device)
    p=torch.sigmoid(logits[:,0:1])
    sem_bce=F.binary_cross_entropy_with_logits(logits[:,0:1],fg)
    sem_dice=1-(2*(p*fg).sum()+1)/(p.sum()+fg.sum()+1)
    sem=sem_bce+sem_dice
    losses=[];weights=[];counts={}
    for axis,name in enumerate(AXES):
        wt=z_weight if axis==0 else 1.
        if wt==0:continue
        target=torch.from_numpy(same[axis].copy()).to(logits.device)
        vm=torch.from_numpy(valid[axis].copy()).to(logits.device)
        lp=logits[0,axis+1]
        positive=vm&(target>0.5)
        negative=vm&(target<0.5)
        npos=int(positive.sum().item());nneg=int(negative.sum().item())
        counts[name]={'same':npos,'different':nneg}
        if npos and nneg:
            loss=.5*(F.softplus(-lp[positive]).mean()+F.softplus(lp[negative]).mean())
        elif npos:loss=F.softplus(-lp[positive]).mean()
        elif nneg:loss=F.softplus(lp[negative]).mean()
        else:continue
        losses.append(loss*wt);weights.append(wt)
    aff=sum(losses)/sum(weights) if losses else logits.sum()*0
    return sem+aff,{'semantic':float(sem.detach()),'affinity':float(aff.detach()),'counts':counts}


def make_network(base, checkpoint:Path):
    model=UNet(dim=3,base=base)
    prior=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if prior.get('model_dim')!=3 or int(prior.get('base',-1))!=base:
        raise ValueError('Checkpoint mismatch: expected 3D U-Net with base '+str(base))
    old=prior['state_dict']
    model.last=nn.Conv3d(base,4,1)
    missing,unexpected=model.load_state_dict(
        {k:v for k,v in old.items() if not k.startswith('last.')},strict=False)
    if set(missing)!={'last.weight','last.bias'} or unexpected:
        raise RuntimeError(f'Unexpected checkpoint state {missing} / {unexpected}')
    with torch.no_grad():
        model.last.weight[0].copy_(old['last.weight'][0]);model.last.bias[0].copy_(old['last.bias'][0])
        nn.init.normal_(model.last.weight[1:],std=.01)
        model.last.bias[1:].zero_()
    return model


def choose_patch(img,ref,tiles,rng,patch_z,patch_xy,guard,min_different_xy=8):
    best=None
    for _ in range(20):
        t=tiles[int(rng.integers(len(tiles)))]
        y0,x0,w=int(t['y']),int(t['x']),int(t['w'])
        if patch_xy>w-2*guard:raise ValueError('Patch XY exceeds inner tile')
        z=int(rng.integers(0,img.shape[0]-patch_z+1))
        y=int(rng.integers(y0+guard,y0+w-guard-patch_xy+1))
        x=int(rng.integers(x0+guard,x0+w-guard-patch_xy+1))
        a=img[z:z+patch_z,y:y+patch_xy,x:x+patch_xy]
        b=ref[z:z+patch_z,y:y+patch_xy,x:x+patch_xy]
        tg,valid=affinity_targets(b)
        n_diff_xy=int(np.logical_and(valid[1:],tg[1:]<.5).sum())
        best=(a,b)
        if n_diff_xy>=min_different_xy and np.mean(b>0)>.02:
            break
    assert best is not None
    return best


def edge_stats(probs,labels):
    same,valid=affinity_targets(labels)
    rows=[]
    for i,name in enumerate(AXES):
        v=valid[i];n=int(v.sum())
        if not n:
            rows.append(dict(axis=name,n_valid=0,balanced_accuracy=None,different_recall=None,different_precision=None,
                             n_same=0,n_different=0))
            continue
        gt_same=same[i][v]>.5
        pred_same=probs[i][v]>=.5
        ns=int(gt_same.sum());nd=int((~gt_same).sum())
        correct_same=int((pred_same&gt_same).sum())
        correct_different=int(((~pred_same)&(~gt_same)).sum())
        pred_diff=int((~pred_same).sum())
        sens_same=correct_same/ns if ns else None
        sens_diff=correct_different/nd if nd else None
        bal=(sens_same+sens_diff)/2 if sens_same is not None and sens_diff is not None else None
        rows.append(dict(axis=name,n_valid=n,n_same=ns,n_different=nd,
                         balanced_accuracy=bal,different_recall=sens_diff,
                         different_precision=correct_different/pred_diff if pred_diff else None,
                         mean_same_prob_on_true_same=float(probs[i][v][gt_same].mean()) if ns else None,
                         mean_same_prob_on_true_different=float(probs[i][v][~gt_same].mean()) if nd else None))
    return rows


@torch.no_grad()
def predict_prob(model,img,device):
    model.eval()
    x=torch.from_numpy(img.copy().astype(np.float32))[None,None].to(device)
    return torch.sigmoid(model(x))[0].cpu().numpy()


def run(args):
    if args.steps<1 or args.patch_z<4 or args.patch_z%2 or args.patch_xy<16 or args.patch_xy%2:
        raise ValueError('Require positive steps, even patch_z >=4, even patch_xy >=16')
    if args.z_weight<0:raise ValueError('z_weight must be nonnegative')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    else:device=args.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    auditfile=args.run_dir/'data_audit.json'
    audit=json.loads(auditfile.read_text())
    if audit.get('stack_order_check',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise RuntimeError('Stack filename order not checked: stop')
    if not args.accept_provisional_registration:
        raise RuntimeError('Must acknowledge PROVISIONAL registration with --accept-provisional-registration')
    img,ref,_,_=read_data(args.data_dir)
    splits=pd.read_csv(args.run_dir/'tile_splits.csv')
    if not {'y','x','w','split'}.issubset(splits.columns):raise ValueError('Malformed tile manifest')
    train=splits.loc[splits['split']=='train'].to_dict('records')
    val=splits.loc[splits['split']=='val'].to_dict('records')
    if not train or not val:raise RuntimeError('No TRAIN/VAL tiles')
    seed=args.seed
    torch.manual_seed(seed);np.random.seed(seed)
    experiments={'xy_only':0.0,'xyz_downweighted':args.z_weight}
    history=[];records=[]
    for name,wz in experiments.items():
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=make_network(args.base,args.run_dir/'unet_3d.pt').to(device)
        optim=torch.optim.AdamW(model.parameters(),lr=args.lr)
        for step in range(1,args.steps+1):
            model.train()
            image_patch,label_patch=choose_patch(img,ref,train,rng,args.patch_z,args.patch_xy,args.guard)
            x=torch.from_numpy(image_patch.copy().astype(np.float32))[None,None].to(device)
            optim.zero_grad(set_to_none=True)
            l,info=conditional_affinity_loss(model(x),label_patch,wz)
            l.backward();optim.step()
            history.append(dict(method=name,step=step,loss=float(l.detach()),**{k:info[k] for k in ('semantic','affinity')}))
            if step==1 or step%50==0 or step==args.steps:
                print(f'{name}: step {step}/{args.steps} loss={float(l.detach()):.4f} sem={info["semantic"]:.4f} aff={info["affinity"]:.4f}',flush=True)
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'base':args.base,'z_weight':wz,'seed':seed,
                    'scope':'TRAIN tiles only, author pseudo labels, single specimen'},args.out/f'{name}.pt')
        for j,t in enumerate(val):
            y,x,w=int(t['y']),int(t['x']),int(t['w'])
            image=img[:,y:y+w,x:x+w]
            lab=ref[:,y:y+w,x:x+w]
            probs=predict_prob(model,image,device)
            g=args.guard
            for metric in edge_stats(probs[1:, :,g:-g,g:-g],lab[:,g:-g,g:-g]):
                records.append(dict(method=name,tile=j,tile_y=y,tile_x=x,**metric))
            print(f'{name}: VAL {j+1}/{len(val)}',flush=True)
        del model
        if device=='cuda':torch.cuda.empty_cache()
    pd.DataFrame(history).to_csv(args.out/'training_history.csv',index=False)
    df=pd.DataFrame(records)
    df.to_csv(args.out/'affinity_validation_tiles.csv',index=False)
    summary=[]
    for (method,axis),block in df.groupby(['method','axis'],sort=False):
        metric=block['balanced_accuracy'].dropna()
        recall=block['different_recall'].dropna()
        summary.append(dict(method=method,axis=axis,n_val_tiles=len(block),
                            median_balanced_accuracy=float(metric.median()) if len(metric) else None,
                            median_different_recall=float(recall.median()) if len(recall) else None,
                            total_valid_pairs=int(block.n_valid.sum()),
                            total_different_pairs=int(block.n_different.sum()),
                            status='NOT_TRAINED' if axis=='Z' and method=='xy_only' else 'SUPERVISED'))
    (args.out/'affinity_learning_validation.json').write_text(json.dumps({
        'scope':'TRAIN optimization, VAL exploratory edge diagnostics, TEST never read',
        'data_source':'Author watershed pseudo-labels, one specimen',
        'patch_zyx':[args.patch_z,args.patch_xy,args.patch_xy],
        'training_steps_per_variant':args.steps,
        'caution':['XY-only means XY affinity SUPERVISION, network still consumes Z context',
                   'Z axis registration and voxel spacing are unverified',
                   'VAL tiles reused during prior development; not independent generalization',
                   'Edge accuracy does not establish instance PQ or cell identity accuracy',
                   'Repeated adjacent edges are not independent biological samples'],
        'summary':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n=== VALIDATION EDGE DIAGNOSTICS ===')
    print(pd.DataFrame(summary).to_string(index=False))
    print('Saved:',args.out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=150)
    p.add_argument('--patch-z',type=int,default=8)
    p.add_argument('--patch-xy',type=int,default=64)
    p.add_argument('--base',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--z-weight',type=float,default=.25)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    p.add_argument('--accept-provisional-registration',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
