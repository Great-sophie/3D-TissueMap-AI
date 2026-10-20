"""Workflow09B: conservative 3D IMC image/mask and author-table correspondence audit.

Never assumes an intensity-table row is a full-mask instance ID.
Does not resample, register, train, or alter images.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

FULL = 'full_mask_final_segmentation_hwatershed_bg500_90%.tif'
MEASURED = 'measured_mask_final_segmentation_hwatershed_bg500_90%.tif'
REGZIP = 'LVILymphBreastCancer_SIMILARITY10_In115.zip'
MODELZIP = 'LVILymphBreastCancerModel.zip'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_key(name: str) -> str:
    base = Path(name).name
    base = re.sub(r'^\d{4}_', '', base)
    base = re.sub(r'__[a-zA-Z0-9]+\.tiff?$', '', base)
    base = re.sub(r'\.ome\.tiff?$', '', base)
    return base


def from_zip_csv(z: ZipFile, short_name: str, **kwargs) -> pd.DataFrame:
    matches = [s for s in z.namelist() if Path(s).name == short_name]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {short_name} in archive, found {len(matches)}')
    with z.open(matches[0]) as f:
        return pd.read_csv(f, **kwargs)


def area_counts(stack: np.ndarray) -> pd.Series:
    vals, counts = np.unique(stack, return_counts=True)
    sel = vals != 0
    return pd.Series(counts[sel].astype(np.int64), index=vals[sel].astype(np.int64), name='voxel_count')


def compare_author_area(author: pd.DataFrame, observed: pd.Series, prefix: str):
    joined = author.merge(observed.rename(f'{prefix}_voxels'), left_on='label', right_index=True, how='left', validate='one_to_one')
    matched = joined[f'{prefix}_voxels'].notna()
    exact = matched & np.isclose(joined['area'], joined[f'{prefix}_voxels'], atol=0, rtol=0)
    stats = {
        'n_mask_ids': int(len(observed)),
        'n_author_ids': int(len(author)),
        'n_shared_ids': int(matched.sum()),
        'n_author_only_ids': int((~matched).sum()),
        'n_mask_only_ids': int(len(observed) - matched.sum()),
        'n_exact_author_area_vs_voxel_count': int(exact.sum()),
        'n_shared_but_area_differs': int((matched & ~exact).sum()),
        'same_id_and_area_mapping_is_confirmed': bool(len(author) == len(observed) and matched.all() and exact.all()),
    }
    return joined[['label',f'{prefix}_voxels']], stats


def mask_outlines_2d(mask: np.ndarray) -> np.ndarray:
    b = np.zeros(mask.shape, dtype=bool)
    b[:,1:] |= (mask[:,1:] != mask[:,:-1]) & (mask[:,1:] > 0)
    b[:,:-1] |= (mask[:,:-1] != mask[:,1:]) & (mask[:,:-1] > 0)
    b[1:,:] |= (mask[1:,:] != mask[:-1,:]) & (mask[1:,:] > 0)
    b[:-1,:] |= (mask[:-1,:] != mask[1:,:]) & (mask[:-1,:] > 0)
    return b


def grayscale_overlay(image: np.ndarray, mask: np.ndarray, lo: float | None = None, hi: float | None = None):
    img = np.log1p(image.astype(np.float32))
    if lo is None: lo = float(np.quantile(img, 0.02))
    if hi is None: hi = float(np.quantile(img, 0.997))
    vals = np.clip((img-lo)/max(hi-lo,1e-6), 0, 1)
    rgb = np.repeat(vals[...,None],3,axis=2)
    ed = mask_outlines_2d(mask)
    rgb[ed] = (1., .22, .17)
    return rgb


def build_figures(stack: np.ndarray, images: list[np.ndarray], output: Path, y: int, x: int, halfwidth: int):
    output.mkdir(parents=True, exist_ok=True)
    n, h, w = stack.shape
    y0, y1 = max(0,y-halfwidth), min(h,y+halfwidth)
    x0, x1 = max(0,x-halfwidth), min(w,x+halfwidth)
    if y1-y0 < 8 or x1-x0 < 8: raise ValueError('Crop is outside the image')
    fig, axs = plt.subplots(4,4,figsize=(16,16),squeeze=False)
    for i, ax in enumerate(axs.flat):
        if i>=n:
            ax.axis('off'); continue
        im = images[i][y0:y1, x0:x1]
        lab = stack[i,y0:y1,x0:x1]
        ax.imshow(grayscale_overlay(im,lab), interpolation='nearest')
        ax.set_title(f'Z index {i}  (slice {i+1})', fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f'Ir193 (grayscale) + full-mask instance borders (red). Fixed XY crop y={y0}:{y1} x={x0}:{x1}\n'
                 'Visual plausibility only; image-to-mask registration NOT certified', fontsize=12)
    fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(output/'ir193_mask_16plane_fixed_roi.png',dpi=140);plt.close(fig)
    choices = sorted(set([0,max(0,n//3),max(0,2*n//3),n-1]))
    fig,axs=plt.subplots(2,2,figsize=(12,12))
    for ax in axs.flat:ax.set_axis_off()
    for ax,i in zip(axs.flat,choices):
        ax.imshow(grayscale_overlay(images[i], stack[i]), interpolation='nearest')
        ax.set_title(f'Ir193 / full-mask Z={i}')
        ax.set_axis_off()
    fig.suptitle('Full field registered intensity TIFFs vs author full 3D mask; alignment unvalidated')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(output/'ir193_mask_fullfield_four_slices.png',dpi=130);plt.close(fig)


def run(args):
    out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Output already exists and not empty: {out}; choose a new --out')
    data_dir=Path(args.data_dir)
    full_path=data_dir/FULL; measured_path=data_dir/MEASURED
    modelzip=Path(args.model_zip); regzip=Path(args.registered_zip)
    for path in (full_path,measured_path,modelzip,regzip):
        if not path.is_file():raise FileNotFoundError(path)
    full=tifffile.imread(full_path); measured=tifffile.imread(measured_path)
    if full.ndim!=3 or measured.ndim!=3 or full.shape[0]!=measured.shape[0]:
        raise ValueError('Expected 3D ZYX masks with equal number of planes')
    full_count=area_counts(full); measured_count=area_counts(measured)
    with ZipFile(modelzip) as z:
        areas=from_zip_csv(z,'LVI_lymph_labels_area.csv')
        means=from_zip_csv(z,'LVI_lymph_mean_intensities.csv')
        order=from_zip_csv(z,'final_3D_stack_order.csv',header=None)
        panel=from_zip_csv(z,'LVI_lymph_panel.csv')
    if not {'label','area'}.issubset(areas.columns) or areas.label.isna().any() or not areas.label.is_unique:
        raise ValueError('Area table missing unique non-null label IDs')
    if (pd.to_numeric(areas.area,errors='coerce').isna()).any():raise ValueError('Non numeric author areas')
    if len(order)!=full.shape[0] or sorted(order.iloc[:,1].astype(int).tolist())!=list(range(full.shape[0])):
        raise ValueError('Author stack-order table inconsistent with mask Z span')
    if 'label' in means.columns: raise ValueError('Intensity table unexpectedly has a label column: investigate before proceeding')
    a,b=compare_author_area(areas,full_count,'full')
    c,d=compare_author_area(areas,measured_count,'measured')
    joined=areas.merge(a,on='label').merge(c,on='label')
    source_keys={int(row.iloc[1]):source_key(str(row.iloc[0])) for _,row in order.iterrows()}
    images=[]; observed=[]
    with ZipFile(regzip) as z:
        names = [n for n in z.namelist() if '/Ir193/' in n and n.endswith('.tiff')]
        if len(names)!=full.shape[0]:raise ValueError(f'Expected {full.shape[0]} registered Ir193 planes, got {len(names)}')
        keyed={}
        for name in names:
            pfx=Path(name).name.split('_')[0]
            if not pfx.isdigit():raise ValueError('Unexpected slice file prefix: '+name)
            idx=int(pfx)-1
            if idx in keyed or idx not in source_keys or source_key(name)!=source_keys[idx]:
                raise ValueError(f'Slice order/name mismatch: {name}')
            keyed[idx]=name
        for zidx in range(full.shape[0]):
            with tifffile.TiffFile(BytesIO(z.read(keyed[zidx]))) as tf:
                im=tf.asarray()
            if im.shape!=full.shape[1:]:raise ValueError(f'Image-mask spatial shape mismatch at Z {zidx}: {im.shape}')
            images.append(im)
            vals=im.astype(float)
            inside=full[zidx]!=0
            observed.append(dict(z_index=zidx, source_filename=keyed[zidx],
                image_shape=str(im.shape), image_dtype=str(im.dtype),
                mask_foreground_pixels=int(inside.sum()),
                image_mean_inside_mask=float(vals[inside].mean()) if inside.any() else None,
                image_mean_outside_mask=float(vals[~inside].mean()) if (~inside).any() else None))
    summary={
        'full_mask_shape':list(full.shape), 'measured_mask_shape':list(measured.shape),
        'full_mask_sha256':digest(full_path),'measured_mask_sha256':digest(measured_path),
        'registered_zip_sha256':digest(regzip), 'model_zip_sha256':digest(modelzip),
        'author_mean_intensity_shape':list(means.shape),
        'author_area_table_shape':list(areas.shape),
        'mean_intensity_has_explicit_instance_id':False,
        'author_area_vs_full_mask':b, 'author_area_vs_measured_mask':d,
        'registration_claim':'NONE: matching TIFF shape/name order does not verify spatial registration',
        'author_intensity_join_claim':'NONE: mean intensity table lacks explicit label mapping',
        'pixel_units_only':True, 'candidate_channel_mappings': panel[panel['Metal Tag'].isin(['Ir193','Cd112','Gd156','Eu151','Er168'])][['Metal Tag','clean_target']].to_dict('records')
    }
    out.mkdir(parents=True,exist_ok=True)
    joined.to_csv(out/'author_area_mask_correspondence.csv',index=False)
    pd.DataFrame(observed).to_csv(out/'ir193_slice_intensity_screen.csv',index=False)
    (out/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    build_figures(full,images,out/'figures',args.crop_y,args.crop_x,args.halfwidth)
    print(json.dumps({
        'mask_instances_full':b['n_mask_ids'], 'mask_instances_measured':d['n_mask_ids'],
        'author_table_rows':len(areas),
        'author_area_matches_full_exact':b['n_exact_author_area_vs_voxel_count'],
        'author_area_matches_measured_exact':d['n_exact_author_area_vs_voxel_count'],
        'author_label_overlap_full':b['n_shared_ids'],'author_label_overlap_measured':d['n_shared_ids'],
        'images':len(images),'out':str(out)
    },indent=2))
    print('CAUTION: no registration or author intensity row-ID mapping certified.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='data/inspect/LVILymph')
    p.add_argument('--model-zip',default='data/raw/LVILymphBreastCancerModel.zip')
    p.add_argument('--registered-zip',default='data/inspect/LVILymph/LVILymphBreastCancer_SIMILARITY10_In115.zip')
    p.add_argument('--out',default='runs/protein_marker_correspondence_audit_v1')
    p.add_argument('--crop-y',type=int,default=384)
    p.add_argument('--crop-x',type=int,default=576)
    p.add_argument('--halfwidth',type=int,default=128)
    args=p.parse_args();run(args)

if __name__=='__main__':main()
