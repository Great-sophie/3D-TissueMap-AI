#!/usr/bin/env python3
"""Cross-slice Identity Review v2: 4-plane registered-coordinate visual audit.

Reference annotations are author pseudo IDs; never used to predict identity.
No transforms, threshold tuning, model inference, or new segmentation.
"""
from __future__ import annotations
import argparse
import csv
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from cross_slice_review_v2_core import (parse_review_ids,center_crop_for_candidate,
                                        common_intensity_limits,mask_outline,plane_indices)

# Separate candidate region outlines and author pseudo-ID outlines.
COLORS = {'pred_A':'#FF4242','pred_B':'#FFBF36',
          'ref_A':'#00E6EA','ref_B':'#DB62F1'}


def as_int(value):
    return int(value) if pd.notna(value) else 0


def draw_outline(ax, binary, color, lw=1.5):
    boundary=mask_outline(binary)
    if boundary.any():
        # contour uses true binary component coordinates, no resampling or shifts
        ax.contour(boundary.astype(float),levels=[0.5],colors=[color],linewidths=lw,
                   origin='upper',extent=(-0.5,boundary.shape[1]-.5,boundary.shape[0]-.5,-.5))


def candidate_figure(image, ref, labels, row, tile_origin_y, tile_origin_x,
                     halfwidth=48, dpi=150):
    z,a,b=(as_int(row[k]) for k in ('z','a','b'))
    za=plane_indices(z,image.shape[0])
    if image.shape!=ref.shape or ref.shape!=labels.shape:raise ValueError('Stack geometry mismatch')
    bounds=center_crop_for_candidate(labels,z,a,b,halfwidth)
    y0,y1,x0,x1=bounds
    vmin,vmax=common_intensity_limits(image,za,bounds)
    ida=as_int(row.get('a_ref_majority',0));idb=as_int(row.get('b_ref_majority',0))
    same_reference_id=ida>0 and ida==idb
    fig,axes=plt.subplots(2,4,figsize=(18,8),squeeze=False,sharex=True,sharey=True)
    for col,zz in enumerate(za):
        for r in range(2):
            ax=axes[r,col]
            if zz<0 or zz>=image.shape[0]:
                ax.set_facecolor('#202020');ax.text(.5,.5,'Outside stack',ha='center',va='center',
                     transform=ax.transAxes,color='white',fontsize=12)
                continue
            ax.imshow(image[zz,y0:y1,x0:x1],cmap='gray',vmin=vmin,vmax=vmax,
                      origin='upper',interpolation='nearest')
            if r==1:
                crop_labels=labels[zz,y0:y1,x0:x1]
                crop_ref=ref[zz,y0:y1,x0:x1]
                if (crop_labels==a).any():draw_outline(ax,crop_labels==a,COLORS['pred_A'],2.)
                if (crop_labels==b).any():draw_outline(ax,crop_labels==b,COLORS['pred_B'],2.)
                if ida>0 and (crop_ref==ida).any():
                    draw_outline(ax,crop_ref==ida,COLORS['ref_A'],1.4)
                if idb>0 and not same_reference_id and (crop_ref==idb).any():
                    draw_outline(ax,crop_ref==idb,COLORS['ref_B'],1.4)
            label = ('A source' if zz==z else 'B target' if zz==z+1 else 'context')
            if r==0:
                ax.set_title(f'Z={zz+1} (1-based) | {label}',fontsize=11,fontweight='bold')
            # Pixel-coordinates of IMC ROI, directly derived from original data layout.
            ticks=np.linspace(0,x1-x0-1,3)
            ticksy=np.linspace(0,y1-y0-1,3)
            ax.set_xticks(ticks,[str(int(round(tile_origin_x+x0+t))) for t in ticks],fontsize=7)
            ax.set_yticks(ticksy,[str(int(round(tile_origin_y+y0+t))) for t in ticksy],fontsize=7)
            ax.set_xlim(-.5,x1-x0-.5);ax.set_ylim(y1-y0-.5,-.5)
        axes[1,col].set_xlabel('Global X (px)',fontsize=9)
    for r in range(2):axes[r,0].set_ylabel(('Ir193 only' if r==0 else 'Ir193 + outlines')+'\nGlobal Y (px)',fontsize=10)
    review_id=str(row['review_id'])
    pz=float(row['mean_pz']) if pd.notna(row.get('mean_pz')) else np.nan
    iou=float(row['footprint_iou']) if pd.notna(row.get('footprint_iou')) else np.nan
    meta=(f'{review_id}   |   pseudo-category: {row.get("reference_category", "unknown")}   |   '
          f'candidate A={a}, B={b}   |   author IDs: A={ida or "unknown"}, B={idb or "unknown"}\n'
          f'Pz={pz:.3f}   footprint IoU={iou:.3f}   |   '
          f'same XY pixel coordinates in all panels; identical Ir193 contrast; NO alignment applied')
    fig.suptitle(meta,fontsize=12,y=.992)
    handles=[Line2D([0],[0],color=COLORS['pred_A'],lw=3,label=f'Predicted A ({a})'),
             Line2D([0],[0],color=COLORS['pred_B'],lw=3,label=f'Predicted B ({b})'),
             Line2D([0],[0],color=COLORS['ref_A'],lw=2,label=f'Author reference A ({ida or "N/A"})')]
    if not same_reference_id:
        handles.append(Line2D([0],[0],color=COLORS['ref_B'],lw=2,
                         label=f'Author reference B ({idb or "N/A"})'))
    else:
        handles.append(Line2D([0],[0],alpha=0,label='Author A/B share the SAME numeric ID'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.024),
               fontsize=10,ncol=4,frameon=False)
    fig.text(.5,.006,'AUTHOR ID IS NOT BIOLOGICAL IDENTITY PROOF   •   Compare nuclear patterns and neighbors across planes',
             ha='center',fontsize=9,color='#9F283A')
    fig.tight_layout(rect=(0,.07,1,.90),w_pad=1.)
    return fig,{'review_id':review_id,'z_zero_based':z,'display_z_1_based':[k+1 if 0<=k<len(image) else None for k in za],
                'candidate_a':a,'candidate_b':b,'reference_a':ida,'reference_b':idb,
                'reference_category':str(row.get('reference_category','unknown')),
                'crop_local_yxyx':[y0,y1,x0,x1],
                'crop_global_yxyx':[tile_origin_y+y0,tile_origin_y+y1,tile_origin_x+x0,tile_origin_x+x1],
                'shared_ir193_display_vmin_vmax':[vmin,vmax]}


def make_index(rows):
    cards=[]
    for r in rows:
        title=html.escape(r['review_id'])
        file=html.escape(r['image_file'],quote=True)
        cat=html.escape(r['reference_category'])
        cards.append(f'<div class="card"><h2>{title} — {cat}</h2><a href="{file}"><img src="{file}" loading="lazy" alt="{title}"></a></div>')
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Cross-slice identity review v2</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:1280px;background:#f6f7f9;color:#18212b}
h1{font-size:1.7rem}.card{background:white;border:1px solid #ddd;border-radius:12px;margin:1.5rem 0;padding:1rem}
.card img{display:block;width:100%;height:auto}.note{background:#e9eef3;border-radius:8px;padding:1rem}</style></head>
<body><h1>Cross-slice Identity Review v2</h1><div class="note">Pseudo-labels are NOT confirmed cell identity.
All panels share XY coordinates and intensity range. Do not infer physical registration from these figures alone.
Fill out <code>manual_z_link_review_v2.csv</code> separately.</div>''' + '\n'.join(cards) + '</body></html>'


def run(args):
    ids=parse_review_ids(args.review_ids)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Output exists and is not empty: {args.out}; choose a new --out')
    if args.halfwidth<4 or args.guard<0:raise ValueError('Invalid halfwidth/guard')
    review_path=args.audit_dir/'manual_z_link_review_template.csv'
    summary_path=args.audit_dir/'z_registration_identity_summary.json'
    if not review_path.exists() or not summary_path.exists():
        raise FileNotFoundError('Need saved Z registration identity audit and its review sheet')
    prior=json.loads(summary_path.read_text())
    tile=prior.get('roi',{})
    y,x=int(tile['tile_y']),int(tile['tile_x'])
    guard=int(tile['guard'])
    if guard!=args.guard:
        raise ValueError(f'Guard mismatch: previous audit={guard}, requested={args.guard}')
    if prior.get('stack_filename_order_validation',{}).get('status')!='FILENAME_ORDER_MATCH':
        raise RuntimeError('Z stack filename order check did not pass')
    from train_segmentation import read_data
    image,reference,_,_=read_data(args.data_dir)
    splits=pd.read_csv(args.split_run_dir/'tile_splits.csv')
    selected=splits[(splits.split=='val')&(splits.y==y)&(splits.x==x)]
    if len(selected)!=1:raise RuntimeError('ROI must be exactly one existing VAL tile')
    w=int(selected.iloc[0].w)
    if w<=2*guard or image.ndim!=3 or image.shape!=reference.shape:
        raise ValueError('Data geometry mismatch')
    ys,xs=y+guard,x+guard
    sl=(slice(None),slice(ys,y+w-guard),slice(xs,x+w-guard))
    ir=image[sl];ref=reference[sl]
    mask_path=args.rescue_run_dir/f'z_link_masks_{y}_{x}.npz'
    with np.load(mask_path,allow_pickle=False) as obj:
        baseline=np.asarray(obj['baseline'])
    if ir.shape!=ref.shape or ir.shape!=baseline.shape or list(ref.shape)!=tile.get('shape_zyx'):
        raise ValueError('Saved baseline mask, audit ROI and Ir193 crop must match')
    review=pd.read_csv(review_path,keep_default_na=False)
    if not {'review_id','z','a','b','a_ref_majority','b_ref_majority','reference_category'}.issubset(review.columns):
        raise ValueError('Review sheet lacks required fields')
    if review.review_id.duplicated().any():raise ValueError('Non-unique review ID')
    missing=set(ids)-set(review.review_id)
    if missing:raise ValueError(f'Review IDs absent from original review sheet: {sorted(missing)}')
    chosen=review.set_index('review_id').loc[ids].reset_index()
    # Strict candidate provenance: verify source z/a/b exist exactly as in saved Z Rescue output.
    source=pd.read_csv(args.rescue_run_dir/f'z_link_candidates_{y}_{x}.csv')
    source_keys=set(map(tuple,source[['z','a','b']].astype(int).to_numpy().tolist()))
    if not all((int(r.z),int(r.a),int(r.b)) in source_keys for r in chosen.itertuples(index=False)):
        raise RuntimeError('Selected candidates do not match original Z-link file')
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for r in chosen.to_dict('records'):
        fig,meta=candidate_figure(ir,ref,baseline,r,ys,xs,args.halfwidth,args.dpi)
        filename=f"{r['review_id']}_{r['reference_category']}_four_plane.png"
        fig.savefig(args.out/filename,dpi=args.dpi,facecolor='white')
        plt.close(fig)
        meta['image_file']=filename
        manifest.append(meta)
        print(f"{r['review_id']}: Z {meta['display_z_1_based']} | reference ({meta['reference_a']}, {meta['reference_b']}) -> {filename}",flush=True)
    frame=chosen.copy()
    frame['v2_manual_identity_assessment']='UNREVIEWED'
    frame['v2_manual_registration_assessment']='UNREVIEWED'
    frame['v2_reviewer_notes']=''
    frame['v2_review_image']=[q['image_file'] for q in manifest]
    frame.to_csv(args.out/'manual_z_link_review_v2.csv',index=False,quoting=csv.QUOTE_MINIMAL)
    (args.out/'review_manifest.json').write_text(json.dumps({'scope':'Existing reused VAL ROI only',
         'reference':'Author watershed pseudo-label (NOT biological identity truth)',
         'candidate_provenance':'Original saved review sheet + Z rescue candidates',
         'no_registration_transform_applied':True,
         'no_model_training_or_test_access':True,
         'review_count':len(manifest),'items':manifest},indent=2,ensure_ascii=False),encoding='utf-8')
    (args.out/'review_gallery.html').write_text(make_index(manifest),encoding='utf-8')
    print('Saved:',args.out)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--split-run-dir',type=Path,required=True)
    p.add_argument('--rescue-run-dir',type=Path,required=True)
    p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--review-ids',default='ZPAIR_004,ZPAIR_006,ZPAIR_024,ZPAIR_026,ZPAIR_041')
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--halfwidth',type=int,default=48)
    p.add_argument('--dpi',type=int,default=150)
    args=p.parse_args()
    run(args)

if __name__=='__main__':main()
