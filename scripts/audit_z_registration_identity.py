#!/usr/bin/env python3
"""Exploratory Z registration, instance-ID continuity, and manual-review atlas.

No image transforms are applied. No model trained or tuned. Reference ID
agreement is pseudo-label consistency, not independently proven cell identity.
Uses ONLY the nominated repeatedly-used development (VAL) ROI for metrics.
"""
from __future__ import annotations
import argparse
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from z_registration_identity_core import (alignment_screen,adjacent_mask_screen,
    id_slice_statistics,stratified_samples,pair_crop_center,crop_bounds,
    image_mask_enrichment_screen)


def source_tiff_tags(data_dir:Path):
    archives=sorted(data_dir.glob('*SIMILARITY*.zip'))
    if len(archives)!=1:return {'status':'NOT_AVAILABLE','reason':'Expected one *SIMILARITY*.zip'}
    from train_segmentation import find_tiff_entries
    with ZipFile(archives[0]) as f:
        entries=find_tiff_entries(f,'Ir193')
        if not entries:return {'status':'NO_IR193'}
        with tifffile.TiffFile(io.BytesIO(f.read(entries[0][1]))) as tif:
            page=tif.pages[0];tags={}
            for k in ['XResolution','YResolution','ResolutionUnit','ImageDescription']:
                if k in page.tags:
                    value=page.tags[k].value
                    tags[k]=str(value)[:1600]
    return {'status':'METADATA_RECORDED_NOT_PHYSICALLY_VALIDATED',
            'archive':archives[0].name,'first_plane':entries[0][1],
            'first_plane_tags':tags,
            'warning':'X/Y TIFF resolution tags are not verified physical pixel size; no Z spacing inferred'}


def axis_breaks_from_reference(ref:np.ndarray):
    rows=[]
    for z in range(ref.shape[0]-1):
        r0=ref[z];r1=ref[z+1]
        both=(r0>0)&(r1>0)
        rows.append({'z':z,'n_both_reference_foreground_faces':int(both.sum()),
                     'n_same_reference_id_faces':int(((r0==r1)&both).sum()),
                     'same_reference_fraction_conditioned_on_both_fg':
                         float(((r0==r1)&both).sum()/both.sum()) if both.any() else None})
    return rows


def make_review_png(review:pd.DataFrame, category:str, image:np.ndarray,reference:np.ndarray,
                    initial:np.ndarray,out:Path,halfwidth:int=40,page_size:int=8):
    subset=review[review.reference_category==category]
    if subset.empty:return
    for page_no,start in enumerate(range(0,len(subset),page_size),start=1):
        sub=subset.iloc[start:start+page_size]
        n=len(sub);fig,axes=plt.subplots(n,2,figsize=(8.4,max(2.5*n,3)),squeeze=False)
        for k,row in enumerate(sub.itertuples(index=False)):
            z=int(row.z);a=int(row.a);b=int(row.b)
            center=pair_crop_center(reference,initial,row)
            y0,y1,x0,x1=crop_bounds(center,reference.shape[1:],halfwidth)
            for j,(zz,pred_id) in enumerate(((z,a),(z+1,b))):
                pic=image[zz,y0:y1,x0:x1]
                # Local relative range, for review visibility only.
                lo,hi=np.percentile(pic,[2,99.5]);gray=np.clip((pic-lo)/(hi-lo+1e-6),0,1)
                rgb=np.repeat(gray[...,None],3,axis=2)
                pred=(initial[zz,y0:y1,x0:x1]==pred_id)
                edge=pred & ~ndi.binary_erosion(pred)
                rgb[edge]=[1.0,0.25,0.15] # red boundary = predicted region
                # Reference majority-ID outline, if available; cyan is reference, not verified cell truth.
                ref_id=int(getattr(row,'a_ref_majority' if j==0 else 'b_ref_majority',0))
                if ref_id>0:
                    re=(reference[zz,y0:y1,x0:x1]==ref_id)
                    boundary=re & ~ndi.binary_erosion(re)
                    rgb[boundary]=[0.5,0.9,1.0] # cyan = author reference
                ax=axes[k,j];ax.imshow(rgb,interpolation='nearest');ax.axis('off')
                ax.set_title(f'{row.review_id} | z={zz+1} | predicted {pred_id}',fontsize=8)
            axes[k,0].set_ylabel(f'Pz={float(row.mean_pz):.2f}\nIoU={float(row.footprint_iou):.2f}',fontsize=8)
        fig.suptitle(f'{category} page {page_no}: red=pred, cyan=author ID; NOT identity proof',fontsize=11)
        fig.tight_layout(rect=[0,0,1,.99])
        fig.savefig(out/f'review_atlas_{category}_page_{page_no:02d}.png',dpi=120)
        plt.close(fig)


def run(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing outputs: {args.out}')
    if not (0<=args.max_shift<=16 and 1<=args.n_per_category<=30):
        raise ValueError('Expected 0<=max_shift<=16 and 1<=n_per_category<=30')
    args.out.mkdir(parents=True,exist_ok=True)
    # Required provenance checks: keep model-free audit within one known VAL ROI.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from train_segmentation import read_data,find_tiff_entries,inspect_stack_order
    tiles=pd.read_csv(args.split_run_dir/'tile_splits.csv')
    rows=tiles[(tiles.split=='val')&(tiles.y==args.tile_y)&(tiles.x==args.tile_x)]
    if len(rows)!=1:raise ValueError('Requested ROI must be exactly one existing VAL tile')
    t=rows.iloc[0];y,x,w,g=int(t.y),int(t.x),int(t.w),args.guard
    if 2*g>=w:raise ValueError('Guard too large')
    im,reference,prefixes,mask_filename=read_data(args.data_dir)
    if y+w>im.shape[1] or x+w>im.shape[2]:raise ValueError('Tile outside data')
    crop=(slice(None),slice(y+g,y+w-g),slice(x+g,x+w-g))
    ir=im[crop];ref=reference[crop]
    if len(ir)!=16:
        raise ValueError(f'Expected examined LVILymph Z=16, got Z={len(ir)}')
    mask_path=args.rescue_run_dir/f'z_link_masks_{y}_{x}.npz'
    with np.load(mask_path) as saved:initial=saved['baseline']
    if initial.shape!=ref.shape:raise ValueError('Saved baseline mask geometry differs from reference crop')
    annotated=pd.read_csv(args.safety_audit_dir/'z_link_candidates_reference_audit.csv')
    if not {'z','a','b','reference_category','a_ref_majority','b_ref_majority','geometry_eligible','mean_pz','footprint_iou'}.issubset(annotated.columns):
        raise ValueError('Reference safety audit missing required columns')
    if (annotated.z<0).any() or (annotated.z>=len(ir)-1).any():raise ValueError('Candidate Z index outside volume')
    raw_path=args.rescue_run_dir/f'z_link_candidates_{y}_{x}.csv'
    if not raw_path.exists():
        raise FileNotFoundError(f'Missing original candidate record for provenance check: {raw_path}')
    raw=pd.read_csv(raw_path)
    key_cols=['z','a','b']
    if any(x not in raw.columns for x in key_cols):raise ValueError('Original candidates missing key columns')
    if len(raw)!=len(annotated) or len(raw.drop_duplicates(key_cols))!=len(raw):
        raise ValueError('Candidate count/uniqueness does not match source rescue run')
    comparison=annotated.merge(raw[key_cols],on=key_cols,how='inner',validate='one_to_one')
    if len(comparison)!=len(raw):
        raise ValueError('Safety audit candidates do not match original rescue run')
    with ZipFile(next(iter(sorted(args.data_dir.glob('*SIMILARITY*.zip'))))) as f:
        entries=find_tiff_entries(f,'Ir193')
    order=inspect_stack_order(args.data_dir/'final_3D_stack_order.csv',entries,len(ir))
    prev_audit=args.split_run_dir/'data_audit.json'
    earlier=json.loads(prev_audit.read_text()).get('stack_order_check') if prev_audit.exists() else None
    z_rows=[]
    for z in range(len(ir)-1):
        a=alignment_screen(ir[z],ir[z+1],args.max_shift)
        b=adjacent_mask_screen(ref[z],ref[z+1])
        a.update({'z':z,'next_z':z+1,**b})
        a['priority_manual_review']=bool(a['peak_at_search_boundary'] or (a['corr_gain']>0.075) or (a['zero_shift_corr']<0.25))
        z_rows.append(a)
    zdf=pd.DataFrame(z_rows)
    zdf.to_csv(args.out/'z_pair_registration_screen.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    axes[0].plot(zdf.z+1,zdf.zero_shift_corr,'o-',label='zero-shift similarity')
    axes[0].plot(zdf.z+1,zdf.best_shift_corr,'s--',label='best bounded-shift similarity')
    axes[0].set_ylabel('Ir193 correlation (heuristic)');axes[0].legend(loc='best')
    axes[1].plot(zdf.z+1,zdf.shift_dy_to_align_next,'o-',label='dy pixels')
    axes[1].plot(zdf.z+1,zdf.shift_dx_to_align_next,'s-',label='dx pixels')
    axes[1].set_ylabel('Suggested bounded shift (px)');axes[1].set_xlabel('Lower plane (1-based)')
    axes[1].legend(loc='best');fig.suptitle('15 adjacent Z pairs: manual registration screen, NOT measured error')
    fig.tight_layout();fig.savefig(args.out/'z_registration_overview.png',dpi=150);plt.close(fig)
    plane_rows=[{'z':z,**image_mask_enrichment_screen(ir[z],ref[z])} for z in range(len(ir))]
    pd.DataFrame(plane_rows).to_csv(args.out/'ir193_mask_alignment_screen.csv',index=False)
    ids=id_slice_statistics(ref)
    ids.to_csv(args.out/'reference_id_slice_continuity.csv',index=False)
    pd.DataFrame(axis_breaks_from_reference(ref)).to_csv(args.out/'reference_same_id_z_faces.csv',index=False)
    review=stratified_samples(annotated,n_per_category=args.n_per_category,seed=args.seed)
    review.to_csv(args.out/'manual_z_link_review_template.csv',index=False)
    for category in ('reliable_same','reliable_different','uncertain'):
        make_review_png(review,category,ir,ref,initial,args.out,halfwidth=args.crop_halfwidth)
    summary={
      'scope':'ONE previously used VAL crop; exploratory 16-layer Z alignment and pseudo-ID audit; no model training or TEST scoring',
      'roi':{'tile_y':y,'tile_x':x,'guard':g,'shape_zyx':list(ref.shape)},
      'stack_filename_order_validation':order,
      'previous_stack_order_check':earlier,
      'tiff_metadata':source_tiff_tags(args.data_dir),
      'mask_filename':mask_filename,'registered_ir193_prefixes':prefixes,
      'registration_screen':{
         'n_adjacent_pairs':len(zdf),
         'n_priority_manual_review':int(zdf.priority_manual_review.sum()),
         'median_abs_shift_px':float(np.median(np.maximum(abs(zdf.shift_dy_to_align_next),abs(zdf.shift_dx_to_align_next)))),
         'median_zero_shift_corr':float(zdf.zero_shift_corr.median()),
         'median_corr_gain':float(zdf.corr_gain.median()),
         'no_transform_applied':True},
      'image_mask_screen':{
         'median_bright_capture_enrichment':float(pd.DataFrame(plane_rows).bright_capture_enrichment_vs_mask_area.median()),
         'no_transform_applied':True,
         'warning':'Ir193 is nuclear-enriched, instance mask is cell-area proxy: overlap is not registration ground truth'},
      'reference_ID_continuity':{
         'n_ids_in_roi':int(len(ids)),
         'n_ids_multiple_slices':int(ids.is_multi_slice.sum()),
         'n_ids_with_nonconsecutive_slice_occupancy':int((ids.n_gaps_in_occupied_slices>0).sum()),
         'n_multi_slice_ids_split_into_6cc':int(((ids.is_multi_slice)&(ids.n_3d_6_connected_components>1)).sum()),
         'n_multi_slice_ids_split_into_26cc':int(((ids.is_multi_slice)&(ids.n_3d_26_connected_components>1)).sum()),
         'note':'Same numeric ID across slices and apparent image alignment do not independently validate single-cell identity'},
      'manual_review':{'n_selected':int(len(review)),
                       'counts':review.reference_category.value_counts().to_dict(),
                       'review_sheet':'manual_z_link_review_template.csv',
                       'manual_labels_allowed':['consistent','inconsistent','uncertain','unreviewed'],
                       'note':'Reference category is a pseudo-label. Do not use it as an expert review result.'},
      'critical_limits':[
          'Intensity-based XY shift screens are NOT registration error estimates and must not be applied blindly.',
          'Image and mask have same pixel dimensions but that does not prove image-to-mask registration.',
          'Filename order verification does not prove physical acquisition order.',
          'Global physical Z spacing and source registration transform are unverified.',
          'Persistent reference instance IDs are author-generated pseudo IDs, not independently validated 3D cell tracking.',
          'All interpretation restricted to one previously viewed VAL crop; do not tune or claim independent generalization.'
       ]}
    (args.out/'z_registration_identity_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Stack-order status:',order['status'])
    print('Z pairs requiring manual registration review:',int(zdf.priority_manual_review.sum()),'/',len(zdf))
    print('ID gaps:',int((ids.n_gaps_in_occupied_slices>0).sum()),'/',len(ids))
    print('Review candidates:',review.reference_category.value_counts().to_dict())
    print('Saved:',args.out)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True,type=Path)
    p.add_argument('--split-run-dir',required=True,type=Path)
    p.add_argument('--rescue-run-dir',required=True,type=Path)
    p.add_argument('--safety-audit-dir',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--tile-y',type=int,default=384)
    p.add_argument('--tile-x',type=int,default=576)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--max-shift',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--n-per-category',type=int,default=20)
    p.add_argument('--crop-halfwidth',type=int,default=40)
    args=p.parse_args();run(args)

if __name__=='__main__':main()
