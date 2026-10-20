#!/usr/bin/env python3
"""3D instance-aware learning from watershed pseudo-labels (exploratory).

Reuse the original 3D U-Net encoder and train 3 output heads:
foreground, instance-contour boundary, and per-instance center. The last two
supervision signals are DERIVED FROM THE SAME AUTHOR WATERSHED MASK and ARE NOT
independent ground truth. Only TRAIN tiles contribute gradients. The four VAL
tiles are used for exploratory comparison; TEST tiles remain untouched.

Usage from project root:
python scripts/train_center_boundary.py --data-dir data/inspect/LVILymph \
  --run-dir runs/segmentation_extended --out runs/center_boundary_learning \
  --steps 200 --patch-xy 64 --patch-z 8 --base 4
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

# Expected to live alongside existing research scripts.
from train_segmentation import UNet, read_data, predict_tile, loss_fn
from instance_watershed_baseline import compare_instances, load_model
from optimize_instance_separation import watershed_instances


def boundary_target(labels: np.ndarray) -> np.ndarray:
    """Inner instance-contour voxels (including cell/background interfaces)."""
    lab = np.asarray(labels)
    if lab.ndim != 3:
        raise ValueError('3D integer instance mask required')
    b = np.zeros(lab.shape, dtype=bool)
    for axis in range(3):
        sl1=[slice(None)]*3
        sl2=[slice(None)]*3
        sl1[axis]=slice(0,-1)
        sl2[axis]=slice(1,None)
        a, c = lab[tuple(sl1)], lab[tuple(sl2)]
        v=(a!=c)
        b[tuple(sl1)] |= v & (a>0)
        b[tuple(sl2)] |= v & (c>0)
    return b.astype(np.float32)


def center_target(labels: np.ndarray, sigma_zyx=(1.0,2.5,2.5)) -> np.ndarray:
    """Gaussian peaks at *cropped pseudo-instance centroids* (not nuclei GT)."""
    lab=np.asarray(labels)
    if lab.ndim != 3:
        raise ValueError('Expected [Z,Y,X] instance labels')
    ids=np.unique(lab)
    ids=ids[ids!=0]
    heat=np.zeros(lab.shape,dtype=np.float32)
    if not len(ids):return heat
    centers=ndi.center_of_mass(lab>0,labels=lab,index=ids.tolist())
    zmax,ymax,xmax=lab.shape
    sig=np.asarray(sigma_zyx,dtype=np.float32)
    if np.any(sig<=0):raise ValueError('Invalid sigma')
    extent=np.ceil(3*sig).astype(int)
    for z,y,x in centers:
        if not np.all(np.isfinite([z,y,x])):continue
        lo=np.maximum(0,np.floor(np.array([z,y,x])-extent).astype(int))
        hi=np.minimum([zmax,ymax,xmax],np.ceil(np.array([z,y,x])+extent+1).astype(int))
        iz,iy,ix=np.ogrid[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
        g=np.exp(-0.5*(((iz-z)/sig[0])**2+((iy-y)/sig[1])**2+((ix-x)/sig[2])**2)).astype(np.float32)
        h=heat[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
        np.maximum(h,g,out=h)
    return heat


def make_model(base: int, old_checkpoint: Path | None = None) -> nn.Module:
    model=UNet(dim=3,base=base)
    # Existing UNet 'last' emits one foreground logit. Extend to three heads.
    old_last=model.last
    model.last=nn.Conv3d(base,3,1)
    with torch.no_grad():
        nn.init.zeros_(model.last.weight)
        nn.init.zeros_(model.last.bias)
    if old_checkpoint is not None:
        # Only load a local model produced by the existing project.
        ck=torch.load(old_checkpoint,map_location='cpu',weights_only=True)
        if ck.get('model_dim')!=3 or int(ck.get('base',-1))!=base:
            raise ValueError('Checkpoint architecture mismatch; match --base')
        initial=ck['state_dict']
        init_without_last={k:v for k,v in initial.items() if not k.startswith('last.')}
        missing,unexpected=model.load_state_dict(init_without_last,strict=False)
        if set(missing)!={'last.weight','last.bias'} or unexpected:
            raise RuntimeError(f'Unexpected checkpoint keys: {missing}, {unexpected}')
        with torch.no_grad():
            model.last.weight[0].copy_(initial['last.weight'][0])
            model.last.bias[0].copy_(initial['last.bias'][0])
            # Auxiliary heads start independent, without importing random content.
            nn.init.normal_(model.last.weight[1:],std=0.01)
            model.last.bias[1:].fill_(-1.5)
    return model


def objective(logits, labels):
    """Semantic BCE+Dice plus inter-instance boundary and center supervision."""
    if logits.shape[1]!=3:raise ValueError('Expected 3 logits channels')
    fg=(labels>0).astype(np.float32)
    boundary=boundary_target(labels)
    centers=center_target(labels)
    device=logits.device
    def t(a):return torch.from_numpy(a.copy())[None,None].to(device)
    sem_loss=loss_fn(logits[:,0:1],t(fg))
    boundary_loss=F.binary_cross_entropy_with_logits(logits[:,1:2],t(boundary),pos_weight=torch.tensor(8.,device=device))
    pred_cent=torch.sigmoid(logits[:,2:3])
    center_true=t(centers)
    center_loss=((pred_cent-center_true)**2*(1+12*center_true)).mean()
    total=sem_loss + 0.25*boundary_loss + 0.7*center_loss
    return total,dict(semantic=float(sem_loss.detach()),boundary=float(boundary_loss.detach()),center=float(center_loss.detach()))


def crop_train(image, ref, tiles, rng, patch_z, patch_xy, guard):
    tile=tiles[int(rng.integers(len(tiles)))]
    y0,x0,w=int(tile['y']),int(tile['x']),int(tile['w'])
    if patch_xy > w - 2*guard:
        raise ValueError('patch-xy too large for train tile')
    z=int(rng.integers(0,image.shape[0]-patch_z+1))
    y=int(rng.integers(y0+guard,y0+w-guard-patch_xy+1))
    x=int(rng.integers(x0+guard,x0+w-guard-patch_xy+1))
    return image[z:z+patch_z,y:y+patch_xy,x:x+patch_xy],ref[z:z+patch_z,y:y+patch_xy,x:x+patch_xy]


def guided_watershed(fg_prob, boundary_prob, center_prob,
                    fg_threshold=.5,center_threshold=.35, min_distance=3, boundary_strength=2.0):
    """Shared semantic mask; learned centers seed watershed; learned boundaries elevate cost."""
    if not (fg_prob.shape==boundary_prob.shape==center_prob.shape) or fg_prob.ndim!=3:
        raise ValueError('Expected matching [Z,Y,X] heads')
    fg=fg_prob>=fg_threshold
    if not fg.any():return np.zeros(fg.shape,np.int32),dict(n_seeds=0,n_fallback=0)
    pts=peak_local_max(ndi.gaussian_filter(center_prob,(0.5,1.,1.)),
                       min_distance=min_distance,threshold_abs=center_threshold,
                       labels=fg,exclude_border=False)
    markers=np.zeros(fg.shape,dtype=np.int32)
    for idx,p in enumerate(pts,1):markers[tuple(p)]=idx
    # Guarantee one seed in every connected foreground component.
    cc,ncc=ndi.label(fg,structure=ndi.generate_binary_structure(3,1))
    present=np.zeros(ncc+1,bool)
    present[cc[markers>0]]=True
    missing=np.flatnonzero(~present[1:])+1
    if len(missing):
        loc=ndi.maximum_position(center_prob,labels=cc,index=missing.tolist())
        if len(missing)==1:loc=[loc]
        for offset,point in enumerate(loc,1): markers[tuple(point)]=len(pts)+offset
    edt=ndi.distance_transform_edt(fg).astype(np.float32)  # voxel distances; spacing unknown
    elevation= -ndi.gaussian_filter(edt,0.8) + boundary_strength*boundary_prob
    result=watershed(elevation,markers,mask=fg)
    return result.astype(np.int32),dict(n_seeds=int(np.count_nonzero(markers)),n_fallback=int(len(missing)))


@torch.no_grad()
def predict_heads(model, input_volume,device):
    v=torch.from_numpy(np.asarray(input_volume,dtype=np.float32).copy())[None,None].to(device)
    model.eval()
    heads=torch.sigmoid(model(v))[0].cpu().numpy().astype(np.float32)
    return heads


def train_and_evaluate(args):
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if args.patch_z<2 or args.patch_z%2 or args.patch_xy%2:
        raise ValueError('3D U-Net expects even patch dimensions')
    np.random.seed(args.seed);torch.manual_seed(args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    img,ref,_,_=read_data(args.data_dir)
    order_path=args.run_dir/'data_audit.json'
    if not order_path.exists():raise FileNotFoundError('Need original data audit')
    audit=json.loads(order_path.read_text())
    if audit.get('stack_order_check',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise RuntimeError('Registered image filename order not verified')
    splits=pd.read_csv(args.run_dir/'tile_splits.csv')
    train=splits.loc[splits['split'].eq('train')].to_dict('records')
    val=splits.loc[splits['split'].eq('val')].to_dict('records')
    if not train or not val:raise RuntimeError('Missing train or validation tiles')
    model=make_model(args.base,args.run_dir/'unet_3d.pt').to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    rng=np.random.default_rng(args.seed)
    hist=[]
    for k in range(1,args.steps+1):
        image_patch,lab_patch=crop_train(img,ref,train,rng,args.patch_z,args.patch_xy,args.guard)
        inp=torch.from_numpy(image_patch.copy())[None,None].to(device)
        model.train();optimizer.zero_grad(set_to_none=True)
        logits=model(inp)
        loss,detail=objective(logits,lab_patch)
        loss.backward();optimizer.step()
        hist.append(dict(step=k,loss=float(loss.detach().cpu()),**detail))
        if k==1 or k%max(1,args.steps//8)==0:
            print(f'progress {k}/{args.steps}: total {hist[-1]["loss"]:.4f}, center {detail["center"]:.4f}',flush=True)
    pd.DataFrame(hist).to_csv(args.out/'train_history.csv',index=False)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},base=args.base,output_heads=['foreground','inner_instance_contour','pseudo_center'],
                   source_checkpoint=str(args.run_dir/'unet_3d.pt'),learning_scope='TRAIN tiles only',seed=args.seed),args.out/'center_boundary_3d.pt')
    rows=[]
    for i,t in enumerate(val,1):
        y,x,w=int(t['y']),int(t['x']),int(t['w'])
        heads=predict_heads(model,img[:,y:y+w,x:x+w],device)
        cut=(slice(None),slice(args.guard,w-args.guard),slice(args.guard,w-args.guard))
        fg,boundary,center=[a[cut] for a in heads]
        rr=ref[:,y+args.guard:y+w-args.guard,x+args.guard:x+w-args.guard]
        # Same model foreground predictions for fair decoder comparison.
        base_pred,_=watershed_instances(fg,min_distance=2,sigma=.8,threshold=.5,ensure_each_component=True)
        guided_pred,stats=guided_watershed(fg,boundary,center,fg_threshold=.5,
                                            center_threshold=args.center_threshold,
                                            min_distance=args.min_distance,
                                            boundary_strength=args.boundary_strength)
        for name,inst in [('distance_same_model',base_pred),('center_boundary_same_model',guided_pred)]:
            metrics=compare_instances(rr,inst,iou_threshold=.5)
            rows.append(dict(tile_y=y,tile_x=x,variant=name,**metrics))
        if i==1:
            z=img.shape[0]//2
            fig,ax=plt.subplots(2,3,figsize=(12,8))
            panels=[img[z,y+args.guard:y+w-args.guard,x+args.guard:x+w-args.guard],rr[z],fg[z],boundary[z],center[z],guided_pred[z]]
            labels=['Ir193','Author pseudo instances','Learned foreground','Learned boundary','Learned center','Learned instance']
            for a,p,label in zip(ax.flat,panels,labels):
                a.imshow(p,cmap='gray' if label=='Ir193' else ('nipy_spectral' if label in ['Author pseudo instances','Learned instance'] else 'viridis'),vmin=0)
                a.set_title(label,fontsize=10);a.axis('off')
            fig.suptitle('Validation-only, one specimen; watershed pseudo-reference',fontsize=11)
            fig.tight_layout();fig.savefig(args.out/'center_boundary_validation.png',dpi=140);plt.close(fig)
        print(f'validation tile {i}/{len(val)} complete',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(args.out/'validation_instance_metrics.csv',index=False)
    rank=[]
    for name,g in df.groupby('variant'):
        rank.append(dict(method=name,n_tiles=len(g),median_pq=float(g.pq.median()),median_recall=float(g.recall.median()),
                         median_precision=float(g.precision.median()),median_predicted=float(g.n_predicted.median())))
    rank.sort(key=lambda d:d['median_pq'],reverse=True)
    report=dict(experiment='Pseudo-label supervised three-head U-Net',scope=f'{len(val)} spatial VAL tiles of one specimen; TEST untouched',
                training_steps=args.steps,train_patch_zyx=[args.patch_z,args.patch_xy,args.patch_xy],
                comparisons='Both postprocessors use exactly the same learned foreground from the same checkpoint',
                limitation=['Author watershed labels are not manual cell truth','No proven physical Z/XY spacing',
                            'Mask-to-Ir193 registration is provisional','Cropped-object PQ is boundary-biased',
                            'Center pseudo targets use crop centroids, not validated nuclear centers',
                            'Validation reuse over many exploratory trials risks overfitting'],
                ranked_validation=rank)
    (args.out/'center_boundary_validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--run-dir',type=Path,default=Path('runs/segmentation_extended'))
    p.add_argument('--out',type=Path,default=Path('runs/center_boundary_learning'))
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--patch-z',type=int,default=8)
    p.add_argument('--patch-xy',type=int,default=64)
    p.add_argument('--base',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--min-distance',type=int,default=3)
    p.add_argument('--center-threshold',type=float,default=.35)
    p.add_argument('--boundary-strength',type=float,default=2.0)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args()
    if args.steps<1 or args.patch_z>16: p.error('Invalid steps or patch-z')
    train_and_evaluate(args)

if __name__=='__main__':main()
