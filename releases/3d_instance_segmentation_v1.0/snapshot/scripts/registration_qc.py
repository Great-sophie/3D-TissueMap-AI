#!/usr/bin/env python3
"""Screen registration of 3D IMC Ir193 signal vs 3D cell instance masks.

This is QC, not a definitive registration validation, and does not infer spatial
micron distances or assume that cell masks should trace nuclear boundaries.

Usage (from project root):
 python scripts/lvilymph_registration_qc.py --data-dir data/inspect/LVILymph \
   --out runs/lvilymph_registration_qc
"""
from __future__ import annotations
import argparse
import io
import json
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile


def normalize(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    good = x[np.isfinite(x)]
    if not good.size:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(good, [1.0, 99.5])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((np.nan_to_num(x, nan=lo) - lo)/(hi-lo), 0, 1)


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    # Boundaries between any different label IDs, without wrapping image edges.
    b = np.zeros(labels.shape, dtype=bool)
    b[1:] |= (labels[1:] != labels[:-1]) & (labels[1:] != 0)
    b[:-1] |= (labels[:-1] != labels[1:]) & (labels[:-1] != 0)
    b[:, 1:] |= (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] != 0)
    b[:, :-1] |= (labels[:, :-1] != labels[:, 1:]) & (labels[:, :-1] != 0)
    return b


def indexed_images(zipfile: ZipFile, metal: str) -> list[tuple[int, str]]:
    found = []
    for f in zipfile.namelist():
        p = PurePosixPath(f)
        if len(p.parts) < 2 or p.parts[-2] != metal or not f.lower().endswith(('.tif','.tiff')):
            continue
        m = re.match(r'^(\d+)_', p.name)
        if not m:
            raise ValueError(f'Slice filename without numeric prefix: {f}')
        found.append((int(m.group(1)), f))
    found.sort()
    if len({i for i, _ in found}) != len(found):
        raise ValueError(f'Duplicate plane indexes in {metal}')
    return found


def bright_mask(img: np.ndarray) -> np.ndarray:
    # Practical, intensity-based proxy for Ir193-rich pixels, not a nucleus label.
    finite = img[np.isfinite(img) & (img > 0)]
    if not finite.size:
        return np.zeros_like(img, dtype=bool)
    threshold = float(np.percentile(finite, 92))
    if threshold <= 0:
        return np.zeros_like(img, dtype=bool)
    return np.isfinite(img) & (img >= threshold)


def shift_capture(mask: np.ndarray, bright: np.ndarray, dy: int, dx: int) -> float:
    H,W = mask.shape
    sy0,sy1=max(0,-dy),min(H,H-dy)
    sx0,sx1=max(0,-dx),min(W,W-dx)
    ty0,ty1=max(0,dy),min(H,H+dy)
    tx0,tx1=max(0,dx),min(W,W+dx)
    if sy1<=sy0 or sx1<=sx0 or not bright.any():
        return float('nan')
    numerator = int((mask[sy0:sy1,sx0:sx1] & bright[ty0:ty1,tx0:tx1]).sum())
    return float(numerator/bright.sum())


def analyze(data_dir: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    masks = list(data_dir.glob('full_mask*.tif'))
    zips = list(data_dir.glob('*SIMILARITY*.zip'))
    if len(masks)!=1 or len(zips)!=1:
        raise FileNotFoundError('Expected exactly one full_mask*.tif and one *SIMILARITY*.zip')
    mask = tifffile.imread(masks[0]);
    if mask.ndim!=3 or mask.dtype.kind not in 'ui':
        raise ValueError(f'Mask should be 3D integer labels; found {mask.shape} {mask.dtype}')
    Z,H,W = mask.shape
    df_order = pd.read_csv(data_dir/'final_3D_stack_order.csv')
    order_preview = df_order.head(20).astype(str).to_dict('records')
    report = {
       'source': 'Kuett LVILymph, real tissue imaging and watershed-derived masks',
       'mask_shape_QYX': list(mask.shape), 'mask_dtype': str(mask.dtype),
       'mask_distinct_nonzero_IDs': int(np.unique(mask[mask!=0]).size),
       'slice_order_columns': df_order.columns.tolist(),
       'slice_order_first20': order_preview,
       'warnings': [
         'Plane mapping by numeric TIFF prefix is provisional; verify final_3D_stack_order.csv with publication code.',
         'This analysis is a spatial plausibility screen, not ground-truth registration accuracy.',
         'Cell instance masks need not hug bright nuclear pixels; masks may expand beyond nuclei.',
         'Proteomic intensity CSV has no explicit label IDs: do not row-wise join without verified metadata.',
         'Voxel size and registered physical Z spacing have NOT been established here.',
         'Do not infer tumor-vessel physical distances or train on image/mask pairs until registration QC passes.'
       ]
    }
    previews: dict[int,tuple[np.ndarray,np.ndarray]]={}
    rows=[]
    with ZipFile(zips[0]) as archive:
        files=indexed_images(archive, 'Ir193')
        report['ir193_plane_prefixes']=[i for i,_ in files]
        if len(files)!=Z:
            raise ValueError(f'Ir193 has {len(files)} images but mask has {Z} planes')
        if len(files)!=len(df_order):
            report['warnings'].append(f'Order CSV has {len(df_order)} rows, TIFF {len(files)}')
        points=set([0, Z//4, Z//2, (3*Z)//4, Z-1])
        for k,(number,name) in enumerate(files):
            arr=tifffile.imread(io.BytesIO(archive.read(name)))
            if arr.shape != (H,W):
                raise ValueError(f'Ir193 plane {number}: TIFF {arr.shape}, mask {(H,W)}')
            fg=(mask[k]!=0)
            bright=bright_mask(arr)
            capture=float((fg & bright).sum()/bright.sum()) if bright.any() else None
            fgfrac=float(fg.mean())
            shifts=[shift_capture(fg,bright,dy,dx) for dy,dx in [(0,150),(0,-150),(150,0),(-150,0)]]
            shift_mean=float(np.nanmean(shifts)) if np.isfinite(shifts).any() else None
            rows.append(dict(plane_zero_based=k, tiff_prefix=number,
                mask_area_fraction=fgfrac, bright_ir193_pixel_fraction=float(bright.mean()),
                ir193_bright_capture_fraction=capture,
                enrichment_vs_mask_coverage=(float(capture/fgfrac) if capture is not None and fgfrac>0 else None),
                shifted_mask_capture_mean=shift_mean,
                note='Proxy only; not registration-error metric'))
            if k in points:
                previews[k]=(normalize(arr),mask[k].copy())
    pd.DataFrame(rows).to_csv(out/'plane_alignment_screen.csv',index=False)
    fig,ax=plt.subplots(2,len(previews),figsize=(3.5*len(previews),7),squeeze=False)
    for col,k in enumerate(sorted(previews)):
        nuc,cellmask=previews[k]
        rgb=np.repeat(nuc[...,None],3,axis=2)
        edge=label_boundaries(cellmask)
        rgb[edge] = [1.0, 0.2, 0.08]
        ax[0,col].imshow(nuc,cmap='gray'); ax[0,col].set_title(f'Ir193, plane {k+1}/{Z}')
        ax[1,col].imshow(rgb); ax[1,col].set_title('Red = cell-label boundary')
        for r in range(2): ax[r,col].axis('off')
    fig.suptitle('REAL 3D IMC — provisional slice-by-slice cell-mask alignment screening')
    fig.tight_layout(); fig.savefig(out/'mask_ir193_multislice.png',dpi=140);plt.close(fig)
    report['registration_screen_by_plane'] = rows
    (out/'registration_screen.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'QC completed: {Z} planes, cell labels from {masks[0].name}')
    print('Saved:',out/'mask_ir193_multislice.png')
    print('Saved:',out/'plane_alignment_screen.csv')
    print('Saved:',out/'registration_screen.json')
    print('\nPlane  | Mask fraction | Bright-Ir193 captured | Shifted-mask captured')
    for r in rows:
        fmt=lambda x: 'NA' if x is None else f'{x:.3f}'
        print(f"{r['plane_zero_based']+1:>5}  | {r['mask_area_fraction']:.3f}         | {fmt(r['ir193_bright_capture_fraction']):>6}                | {fmt(r['shifted_mask_capture_mean']):>6}")
    print('\nIMPORTANT: This is QC screening only; compare stack order and registration metadata before modeling.')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    ap.add_argument('--out',type=Path,default=Path('runs/lvilymph_registration_qc'))
    args=ap.parse_args(); analyze(args.data_dir,args.out)

if __name__=='__main__': main()
