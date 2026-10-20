#!/usr/bin/env python3
"""Exploratory 2D-versus-3D IMC cell-foreground learning.

This script uses a *single* tissue volume and watershed-derived pseudo-reference labels.
Spatially separated XY tiles are NOT independent samples/patients. It does not
validate 3D cell instances, 3D morphology, or biological effects.

Usage from project root:
  python scripts/train_segmentation.py --data-dir data/inspect/LVILymph \
      --out runs/segmentation_baselines --epochs 3 --steps 60
"""
from __future__ import annotations
import argparse
import io
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F


def find_tiff_entries(archive, metal):
    found = []
    for path in archive.namelist():
        p = PurePosixPath(path)
        if len(p.parts) < 2 or p.parts[-2] != metal or p.suffix.lower() not in ('.tif', '.tiff'):
            continue
        match = re.match(r'^(\d+)_', p.name)
        if not match:
            raise ValueError('Cannot infer index from TIFF name: ' + path)
        found.append((int(match.group(1)), path))
    found.sort(key=lambda x: x[0])
    if len({n for n,_ in found}) != len(found):
        raise ValueError('Duplicate slice indices in channel ' + metal)
    return found


def normalize_slice(a):
    a = np.asarray(a, dtype=np.float32)
    pos = a[np.isfinite(a) & (a > 0)]
    if pos.size < 50:
        return np.zeros_like(a, np.float32)
    lo, hi = np.percentile(pos, [1, 99.5])
    if hi <= lo:
        return np.zeros_like(a, np.float32)
    return np.clip((np.nan_to_num(a, nan=0., posinf=0., neginf=0.) - float(lo)) / float(hi-lo), 0, 1).astype(np.float32)


def read_data(data_dir: Path):
    masks = sorted(data_dir.glob('full_mask*.tif'))
    archives = sorted(data_dir.glob('*SIMILARITY*.zip'))
    if len(masks) != 1 or len(archives) != 1:
        raise FileNotFoundError('Need one full_mask*.tif and one *SIMILARITY*.zip in ' + str(data_dir))
    lab = tifffile.imread(masks[0])
    if lab.ndim != 3 or lab.dtype.kind not in 'ui':
        raise ValueError('3D uint instance labels expected, got ' + str((lab.shape,lab.dtype)))
    z_n,h,w = lab.shape
    frames = []
    with ZipFile(archives[0]) as zipf:
        entries = find_tiff_entries(zipf, 'Ir193')
        if len(entries) != z_n:
            raise ValueError(f'Ir193 slices {len(entries)} != mask planes {z_n}')
        for n,path in entries:
            with zipf.open(path) as f:
                img = tifffile.imread(io.BytesIO(f.read()))
            if img.shape != (h,w):
                raise ValueError(f'Ir193 plane {n} shape {img.shape} != mask {(h,w)}')
            frames.append(normalize_slice(img))
    im = np.stack(frames).astype(np.float32)
    return im, lab, [n for n,_ in entries], masks[0].name



def canonical_section_filename(name):
    """Match an author's original OME filename to a registered marker TIFF.

    Example:
      0007_20190220_..._r7_a7_ac__Ir193.tiff
      20190220_..._r7_a7_ac.ome.tiff
    """
    name = PurePosixPath(str(name).replace('\\', '/')).name.strip()
    name = re.sub(r'^\d{4}_', '', name)
    name = re.sub(r'(?i)\.ome\.tiff?$', '', name)
    name = re.sub(r'(?i)\.tiff?$', '', name)
    name = re.sub(r'__[A-Za-z][A-Za-z0-9_-]*$', '', name)
    return name


def inspect_stack_order(csv_path, numbered_entries, expected_planes):
    """Independent filename-based check of the declared stack order.

    TIFF prefixes are 1-based in the inspected LVILymph series. The CSV
    numeric column may be zero-based or one-based; we infer a constant
    offset only if every original filename agrees with its registered TIFF.
    """
    result = dict(status='UNVERIFIED', n_expected=expected_planes,
                  warning='Filename agreement does not validate physical slice-to-mask registration')
    if not Path(csv_path).exists():
        result['reason']='stack-order CSV is missing'
        return result
    try:
        df=pd.read_csv(csv_path, header=None, names=['original_filename','stack_index'], dtype=str)
        df=df.dropna(how='all')
        if len(df)!=expected_planes:
            raise ValueError(f'CSV records {len(df)} != image planes {expected_planes}')
        if df[['original_filename','stack_index']].isna().any().any():
            raise ValueError('CSV contains missing filename/order values')
        order=pd.to_numeric(df['stack_index'],errors='raise').to_numpy()
        if not np.all(np.equal(order,np.floor(order))):
            raise ValueError('stack indices are not integers')
        order=order.astype(int)
        if len(np.unique(order))!=expected_planes:
            raise ValueError('duplicate stack indices')
        indexed={canonical_section_filename(path):prefix for prefix,path in numbered_entries}
        if len(indexed)!=expected_planes:
            raise ValueError('registered TIFF filenames are not unique after normalization')
        originals=[canonical_section_filename(a) for a in df.original_filename]
        if len(set(originals))!=expected_planes:
            raise ValueError('duplicate original section filenames')
        if set(indexed)!=set(originals):
            missing=sorted(set(indexed)-set(originals))[:4]
            extra=sorted(set(originals)-set(indexed))[:4]
            raise ValueError(f'CSV/TIFF original filename mismatch: missing={missing}; extra={extra}')
        offsets=[indexed[key]-int(z) for key,z in zip(originals,order)]
        if len(set(offsets))!=1 or offsets[0] not in (0,1):
            raise ValueError(f'CSV order conflicts with registered TIFF prefix (offsets: {sorted(set(offsets))})')
        if sorted(indexed.values())!=list(range(1,expected_planes+1)):
            raise ValueError('registered image prefixes are not exactly 1..N')
        if sorted(order.tolist())!=list(range(1-offsets[0],expected_planes+1-offsets[0])):
            raise ValueError('CSV order indices do not cover the expected sequence')
        result.update(status='FILENAME_ORDER_MATCH',csv_rows=len(df),
                      csv_index_base=1-offsets[0],registered_prefix_range=[1,expected_planes],
                      filename_matches=expected_planes,
                      sample_mappings=[dict(original_filename=str(df.iloc[i,0]),csv_index=int(order[i]),
                                            registered_prefix=indexed[originals[i]]) for i in range(min(3,len(df)))])
    except Exception as e:
        result.update(status='BLOCKED', reason=str(e))
    return result

def instance_audit(lab):
    z = lab.shape[0]
    ratios = []
    id_sets = []
    for k in range(z):
        a = set(np.unique(lab[k]).tolist())
        a.discard(0)
        id_sets.append(a)
    for k in range(z-1):
        a,b = lab[k],lab[k+1]
        overlap = (a>0) & (b>0)
        ratios.append({'pair':[k,k+1], 'overlapping_foreground_voxels':int(overlap.sum()),
                       'same_ID_fraction_where_both_foreground':float(np.mean(a[overlap]==b[overlap])) if overlap.any() else None,
                       'shared_instance_IDs':len(id_sets[k] & id_sets[k+1])})
    ids_union = set().union(*id_sets)
    multi = sum(sum(label in ids for ids in id_sets)>1 for label in ids_union)
    return {'mask_shape_ZYX':list(lab.shape), 'global_instance_ID_count':len(ids_union),
            'fraction_IDs_seen_in_multiple_slices':float(multi/len(ids_union)) if ids_union else None,
            'adjacent_plane_ID_continuity':ratios,
            'caution':'Persistent IDs are necessary but not sufficient for valid 3D instances.'}


def spatial_tiles(lab, tile, guard, seed, min_fg=0.002):
    # Splits a single tissue *spatially* with guard bands. Not independent validation.
    _,H,W = lab.shape
    if tile < 2*guard + 32 or tile % 4:
        raise ValueError('tile must be divisible by 4 and > 2*guard+32')
    all_tiles=[]
    for y in range(0, H-tile+1, tile):
        for x in range(0, W-tile+1, tile):
            fg=float(np.mean(lab[:,y+guard:y+tile-guard,x+guard:x+tile-guard]>0))
            all_tiles.append(dict(y=y,x=x,h=tile,w=tile,fg_fraction=fg,eligible=fg>=min_fg))
    eligible=[i for i,t in enumerate(all_tiles) if t['eligible']]
    if len(eligible)<6:
        raise ValueError(f'Only {len(eligible)} eligible tissue tiles. Try --tile 160 or --min-fg 0.0005.')
    rng=np.random.default_rng(seed)
    idx=rng.permutation(eligible)
    ntest=max(1,int(round(len(idx)*0.2)))
    nval=max(1,int(round(len(idx)*0.2)))
    for t in all_tiles: t['split']='excluded_background'
    for i in idx[:ntest]: all_tiles[int(i)]['split']='test'
    for i in idx[ntest:ntest+nval]: all_tiles[int(i)]['split']='val'
    for i in idx[ntest+nval:]: all_tiles[int(i)]['split']='train'
    if not any(t['split']=='train' for t in all_tiles):
        raise ValueError('No training tiles')
    for t in all_tiles:
        t['x1']=t['x']+t['w'];t['y1']=t['y']+t['h']
    return all_tiles


class Block(nn.Module):
    def __init__(self, cin, cout, dim):
        super().__init__()
        conv=nn.Conv3d if dim==3 else nn.Conv2d
        norm=nn.InstanceNorm3d if dim==3 else nn.InstanceNorm2d
        self.seq=nn.Sequential(conv(cin,cout,3,padding=1),norm(cout,affine=True),nn.LeakyReLU(0.1,inplace=True),
                               conv(cout,cout,3,padding=1),norm(cout,affine=True),nn.LeakyReLU(0.1,inplace=True))
    def forward(self,x): return self.seq(x)


class UNet(nn.Module):
    def __init__(self,dim=3,base=8):
        super().__init__()
        pool=nn.MaxPool3d if dim==3 else nn.MaxPool2d
        transpose=nn.ConvTranspose3d if dim==3 else nn.ConvTranspose2d
        conv=nn.Conv3d if dim==3 else nn.Conv2d
        self.enc1=Block(1,base,dim)
        self.enc2=Block(base,base*2,dim)
        self.pool=pool(2)
        self.up=transpose(base*2,base,2,stride=2)
        self.dec=Block(base*2,base,dim)
        self.last=conv(base,1,1)
    def forward(self,x):
        a=self.enc1(x)
        b=self.enc2(self.pool(a))
        up=self.up(b)
        if up.shape[2:]!=a.shape[2:]: raise ValueError('Spatial sizes must be even')
        return self.last(self.dec(torch.cat([a,up],dim=1)))


def loss_fn(logits,truth):
    bce=F.binary_cross_entropy_with_logits(logits,truth)
    p=torch.sigmoid(logits)
    dims=tuple(range(1,p.ndim))
    dice=(2*(p*truth).sum(dims)+1)/(p.sum(dims)+truth.sum(dims)+1)
    return bce + (1-dice).mean()


def random_crop(data,lab,tile,dim,patch_z,patch_xy,guard,rng):
    y0,x0=tile['y'],tile['x']; full=tile['w']
    if patch_xy>full-2*guard: raise ValueError(f'Patch XY {patch_xy} cannot fit tile {full} guard {guard}')
    y=int(rng.integers(y0+guard,y0+full-guard-patch_xy+1))
    x=int(rng.integers(x0+guard,x0+full-guard-patch_xy+1))
    Z=data.shape[0]
    if dim==3:
        if patch_z>Z or patch_z%2: raise ValueError('patch-z must be even and <= number of slices')
        z=int(rng.integers(0,Z-patch_z+1))
        a=data[z:z+patch_z,y:y+patch_xy,x:x+patch_xy]
        b=lab[z:z+patch_z,y:y+patch_xy,x:x+patch_xy]>0
    else:
        z=int(rng.integers(0,Z))
        a=data[z,y:y+patch_xy,x:x+patch_xy]
        b=lab[z,y:y+patch_xy,x:x+patch_xy]>0
    return torch.from_numpy(a.copy())[None,None],torch.from_numpy(b.astype(np.float32))[None,None]


def train(model,dim,image,mask,tiles,device,epochs,steps,patch_z,patch_xy,guard,seed,out):
    rng=np.random.default_rng(seed+dim)
    candidates=[t for t in tiles if t['split']=='train']
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    history=[]
    for epoch in range(1,epochs+1):
        model.train()
        losses=[]
        for _ in range(steps):
            tile=candidates[int(rng.integers(len(candidates)))]
            a,b=random_crop(image,mask,tile,dim,patch_z,patch_xy,guard,rng)
            a,b=a.to(device),b.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred=model(a)
            loss=loss_fn(pred,b)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({'epoch':epoch,'mean_train_loss':float(np.mean(losses))})
        print(f'  {dim}D epoch {epoch}/{epochs}: train loss {np.mean(losses):.4f}',flush=True)
    torch.save({'model_dim':dim,'base':model.enc1.seq[0].out_channels,'state_dict':model.cpu().state_dict(),
                'source':'watershed-derived reference; one specimen, not ground truth',
                'seed':seed,'training_tile_count':len(candidates)},out/f'unet_{dim}d.pt')
    return history


@torch.no_grad()
def predict_tile(model,dim,image,tile,device):
    # Entire XY tile, no neighboring pixels; all Z levels for fair tissue region.
    y,x=tile['y'],tile['x']; n=tile['w']
    x0=image[:,y:y+n,x:x+n]
    model.eval()
    if dim==2:
        outs=[]
        for z in range(len(x0)):
            a=torch.from_numpy(x0[z].copy())[None,None].to(device)
            outs.append(torch.sigmoid(model(a))[0,0].cpu().numpy())
        return np.stack(outs)
    a=torch.from_numpy(x0.copy())[None,None].to(device)
    try:
        return torch.sigmoid(model(a))[0,0].cpu().numpy()
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError('Inference GPU OOM; rerun with --tile 160 or --base 4') from e


def metrics(prob,truth,guard=0):
    if guard: prob=prob[:,guard:-guard,guard:-guard];truth=truth[:,guard:-guard,guard:-guard]
    y=truth>0;p=prob>=.5
    tp=int(np.logical_and(p,y).sum());fp=int(np.logical_and(p,~y).sum());fn=int(np.logical_and(~p,y).sum())
    dice=(2*tp+1)/(2*tp+fp+fn+1)
    iou=(tp+1)/(tp+fp+fn+1)
    return {'dice':float(dice),'iou':float(iou),'tp':tp,'fp':fp,'fn':fn,'true_fg_fraction':float(y.mean()),'pred_fg_fraction':float(p.mean())}


def show_example(im,truth,p2,p3,where,out):
    z=im.shape[0]//2
    fig, ax=plt.subplots(2,2,figsize=(9,9))
    ax[0,0].imshow(im[z],cmap='gray');ax[0,0].set_title('Ir193 (real image)')
    ax[0,1].imshow(truth[z]>0,cmap='gray');ax[0,1].set_title('Watershed reference (NOT manual GT)')
    ax[1,0].imshow(p2[z],cmap='viridis',vmin=0,vmax=1);ax[1,0].set_title('2D U-Net probability')
    ax[1,1].imshow(p3[z],cmap='viridis',vmin=0,vmax=1);ax[1,1].set_title('3D U-Net probability')
    for a in ax.flat:a.axis('off')
    fig.suptitle(f'HELD-OUT XY TILE {where}: same specimen, preliminary',fontsize=11)
    fig.tight_layout();fig.savefig(out/'prediction_comparison.png',dpi=125);plt.close(fig)


def build_instance_preview(proba,out):
    """Unvalidated watershed-style partition of predicted semantic foreground.
    Requires scikit-image; no reference instance metrics are claimed.
    """
    try:
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
    except ImportError:
        return {'status':'skipped','reason':'install scikit-image for exploratory watershed'}
    fg=proba>=.5
    if not fg.any():
        return {'status':'skipped','reason':'no foreground at threshold 0.5'}
    edt=ndi.distance_transform_edt(fg)
    pts=peak_local_max(edt,min_distance=3,threshold_abs=1,labels=fg,exclude_border=False)
    seeds=np.zeros(fg.shape,dtype=np.int32)
    for i,(z,y,x) in enumerate(pts,1):seeds[z,y,x]=i
    if not len(pts):return {'status':'skipped','reason':'no seeds'}
    watershed_labels=watershed(-edt,markers=seeds,mask=fg).astype(np.int32)
    np.savez_compressed(out/'predicted_instances_provisional.npz',labels=watershed_labels)
    return {'status':'provisional_only', 'n_predicted_objects':int(np.unique(watershed_labels[watershed_labels>0]).size),
            'warning':'Naive isotropic voxel-space watershed; NOT a validated 3D instance segmentation. No physical Z spacing or biological claim.'}


def run(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    else: device=args.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but not available')
    print('Device:',device,flush=True)
    im,inst,indices,maskname=read_data(args.data_dir)
    print('Ir193 stack:',im.shape,'mask:',inst.shape,'plane indices:',indices,flush=True)
    audit=instance_audit(inst)
    order_file=args.data_dir/'final_3D_stack_order.csv'
    with ZipFile(next(iter(sorted(args.data_dir.glob('*SIMILARITY*.zip'))))) as zf:
        entries=find_tiff_entries(zf, 'Ir193')
    order_check=inspect_stack_order(order_file, entries, im.shape[0])
    audit['stack_order_check']=order_check
    print('Stack-order check:',json.dumps(order_check,ensure_ascii=False),flush=True)
    intensity_file=args.data_dir/'LVI_lymph_mean_intensities.csv'
    area_file=args.data_dir/'LVI_lymph_labels_area.csv'
    if intensity_file.exists() and area_file.exists():
        area=pd.read_csv(area_file)
        df=pd.read_csv(intensity_file,nrows=2)
        audit['cell_tables']={'label_area_rows':len(area),
             'intensity_has_label_id':'label' in df.columns,
             'mapping_to_reference_mask':'NOT VERIFIED: never pair rows without provenance'}
    audit.update({'real_data':True,'mask_filename':maskname,'image_channel':'Ir193',
      'image_mask_registration':'PROVISIONAL; verified metadata correspondence still required',
      'slice_order':'filename-level CSV audit: '+order_check['status'],
      'reference_type':'author-provided watershed masks, NOT independent manual ground truth',
      'physical_voxel_spacing_um':'UNKNOWN; must be verified, no physical distances computed',
      'protein_intensities_join':'NOT ATTEMPTED: intensity table lacks verified label mapping'})
    (args.out/'data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Fraction of IDs across >1 slice:',audit['fraction_IDs_seen_in_multiple_slices'])
    # Save reproducible tile partitions; train/val/test are spatially disjoint, but same specimen.
    tiles=spatial_tiles(inst,args.tile,args.guard,args.seed,args.min_fg)
    pd.DataFrame(tiles).to_csv(args.out/'tile_splits.csv',index=False)
    print('Tile splits:',dict(Counter(t['split'] for t in tiles)),flush=True)
    if args.audit_only:
        print('AUDIT ONLY; modeling not run. Outputs:',args.out)
        return
    if order_check['status'] != 'FILENAME_ORDER_MATCH':
        raise RuntimeError('STOP: stack-order CSV and registered TIFF filenames not fully matched. See data_audit.json.')
    if not args.accept_provisional_registration:
        raise RuntimeError('STOP: alignment with the provided mask is only provisional. For EXPLORATORY learning, rerun with --accept-provisional-registration, after reviewing overlays and registration QC.')
    if args.patch_xy%2 or args.patch_z%2 or args.tile%2:
        raise ValueError('patch and tile dimensions must be even')
    if args.patch_xy>args.tile-2*args.guard:
        raise ValueError('patch-xy > tile - 2*guard, reduce patch-xy or increase tile')
    history={}
    outputs={}
    for dim in (2,3):
        model=UNet(dim,args.base).to(device)
        print(f'Training {dim}D model...',flush=True)
        history[str(dim)]=train(model,dim,im,inst,tiles,device,args.epochs,args.steps,args.patch_z,
                               args.patch_xy,args.guard,args.seed,args.out)
        model.to(device).eval()
        rows=[];saved=None
        for t in (r for r in tiles if r['split'] in ['val','test']):
            with torch.no_grad(): pred=predict_tile(model,dim,im,t,device)
            z,y,x=im.shape[0],t['y'],t['x'];w=t['w']
            truth=inst[:,y:y+w,x:x+w]
            m=metrics(pred,truth,args.guard)
            rows.append(dict(model=f'{dim}D',split=t['split'],tile_y=y,tile_x=x,**m))
            if t['split']=='test' and saved is None:
                saved=(t,pred)
        outputs[dim]={'rows':rows,'saved':saved}
        pd.DataFrame(rows).to_csv(args.out/f'metrics_{dim}d.csv',index=False)
        if device=='cuda':torch.cuda.empty_cache()
    if outputs[2]['saved'] and outputs[3]['saved']:
        t=outputs[3]['saved'][0]
        assert t['x']==outputs[2]['saved'][0]['x'] and t['y']==outputs[2]['saved'][0]['y']
        y,x=t['y'],t['x'];n=t['w']
        img=im[:,y:y+n,x:x+n];lab=inst[:,y:y+n,x:x+n]
        p2=outputs[2]['saved'][1];p3=outputs[3]['saved'][1]
        show_example(img,lab,p2,p3,f'Y{y} X{x}',args.out)
        np.savez_compressed(args.out/'example_predictions.npz',reference_binary=(lab>0).astype(np.uint8),
                            p2d=p2.astype(np.float16),p3d=p3.astype(np.float16))
        provisional=build_instance_preview(p3,args.out)
    else:provisional={'status':'not_run'}
    summary={'data':'REAL Kuett LVILymph IMC', 'volume_shape':list(inst.shape),
             'study_design':'Within-one-volume spatial holdout, exploratory only; NO independent patient generalization',
             'image_mask_alignment':'PROVISIONAL, not fully verified',
             'stack_order':'PROVISIONAL, numeric prefixes',
             'voxel_spacing':'UNVERIFIED', 'label_source':'Watershed pseudo-reference labels',
             'learning_task':'cell foreground binary semantic segmentation from Ir193 (NOT histological phenotype identification)',
             'tile_counts':dict(Counter(t['split'] for t in tiles)),
             'training_parameters':{k:getattr(args,k) for k in ['epochs','steps','tile','guard','patch_z','patch_xy','base','seed']},
             'history':history,
             'metrics':{},'instance_preview':provisional}
    for dim in (2,3):
        rows=[r for r in outputs[dim]['rows'] if r['split']=='test']
        tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
        summary['metrics'][f'{dim}D_test']={'pooled_dice':(2*tp+1)/(2*tp+fp+fn+1),
                                           'pooled_iou':(tp+1)/(tp+fp+fn+1),
                                           'num_test_tiles':len(rows),
                                           'interpretation':'pseudo-label agreement, not generalization'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(6,4))
    for dim in (2,3):
        vals=history[str(dim)]
        ax.plot([v['epoch'] for v in vals],[v['mean_train_loss'] for v in vals],marker='o',label=f'{dim}D')
    ax.set(xlabel='Epoch',ylabel='Train loss',title='Exploratory same-specimen models')
    ax.legend();fig.tight_layout();fig.savefig(args.out/'training_curves.png',dpi=130);plt.close(fig)
    print(json.dumps(summary['metrics'],indent=2),flush=True)
    print('Outputs saved:',args.out)
    print('CAUTION: numbers measure agreement with existing watershed labels in one tissue. Not independent validation.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--out',type=Path,default=Path('runs/segmentation_baselines'))
    p.add_argument('--audit-only',action='store_true',help='Read data and split tiles, no training')
    p.add_argument('--accept-provisional-registration',action='store_true',help='Allow exploratory same-specimen training after inspecting QC; NOT a claim of validated registration')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--tile',type=int,default=192)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--min-fg',type=float,default=.002)
    p.add_argument('--patch-xy',type=int,default=128)
    p.add_argument('--patch-z',type=int,default=8)
    p.add_argument('--base',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    if a.epochs<1 or a.steps<1:raise ValueError('--epochs and --steps must be positive')
    run(a)

if __name__=='__main__':main()
