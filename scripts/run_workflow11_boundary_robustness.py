#!/usr/bin/env python3
"""WF11D v0.3 exploratory: small XY shift and Voronoi competition-boundary ablation.

Compatible with the frozen WF09 baseline and pre-existing WF11 v0.1/v0.2
scripts. Per-section 2D XY; does NOT identify cell membranes or physical 3D.
"""
from __future__ import annotations
import argparse
import json
import shutil
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_signal_attribution_core import REGIONS, assign_slice
from run_spatial_signal_attribution import sha, file_manifest
from run_workflow11_shift_controls import (
    ensure_run_integrity, check_zero_shift_matches_wf11,
    translated_no_wrap, fast_bin, summarize_contrast, group_ids,
    COMPARISONS, DEFAULT_MARKERS,
)

MICRO_OFFSETS = ((0,0), (0,1), (0,-1), (1,0), (-1,0),
                 (0,2), (0,-2), (2,0), (-2,0),
                 (0,4), (0,-4), (4,0), (-4,0))
MARGINS = (0,1,2)


def competition_boundary(owners: np.ndarray) -> np.ndarray:
    """Pixels touching 4-neighbor ownership changes; 0 is unlabeled/not compared."""
    owners = np.asarray(owners)
    if owners.ndim!=2:
        raise ValueError('2D ownership map required')
    b = np.zeros(owners.shape, dtype=bool)
    horiz = (owners[:,:-1] > 0) & (owners[:,1:] > 0) & (owners[:,:-1] != owners[:,1:])
    b[:,:-1] |= horiz
    b[:,1:] |= horiz
    vert = (owners[:-1,:] > 0) & (owners[1:,:] > 0) & (owners[:-1,:] != owners[1:,:])
    b[:-1,:] |= vert
    b[1:,:] |= vert
    return b


def cached_geometry(mask_slice: np.ndarray, nids: int):
    """Original region/owner assignment; margin filters exclude only disputed rings.

    Excluding nearby Voronoi-owner *boundaries* is a contamination sensitivity
    PROXY, not a truth-labelled neighbor-cell mask or verified cell boundary.
    """
    own = assign_slice(mask_slice, nids)
    selected, codes = own.selected, own.codes
    owner_map=np.zeros(mask_slice.shape, dtype=np.int32)
    owner_map.ravel()[selected]=own.owners
    boundary=competition_boundary(owner_map)
    nearest=(distance_transform_edt(~boundary) if boundary.any()
             else np.full(mask_slice.shape, np.inf))
    # Never discard nuclear reference voxels. Only remove shell pixels near
    # differing Voronoi ownership domains, at distance <= 1/2 px.
    is_nuclear=(own.regions.ravel()[selected] == 0)
    distance=nearest.ravel()[selected]
    subsets = []
    for margin in MARGINS:
        keep = np.ones(len(selected), dtype=bool) if margin == 0 else (is_nuclear | (distance > margin))
        subsets.append((selected[keep], codes[keep]))
    return subsets, int(boundary.sum()), int(len(selected)), [int(len(i)) for i,_ in subsets]


def accumulate_marker(mask, paths, nids, cache):
    shape=(len(MICRO_OFFSETS),len(REGIONS),nids+1)
    sums=np.zeros(shape, dtype=np.float64)
    counts=np.zeros(shape, dtype=np.int64)
    margin_sums=np.zeros((len(MARGINS),len(REGIONS),nids+1),dtype=np.float64)
    margin_counts=np.zeros_like(margin_sums,dtype=np.int64)
    for z in range(mask.shape[0]):
        image=tifffile.imread(paths[z])
        if image.shape != mask.shape[1:]:
            raise ValueError(f'Section {z}: channel geometry mismatch')
        subsets=cache[z][0]
        for j, (dy,dx) in enumerate(MICRO_OFFSETS):
            a=translated_no_wrap(image, dy, dx)
            total,count=fast_bin(subsets[0],a,nids)
            sums[j]+=total
            counts[j]+=count
        for k in range(len(MARGINS)):
            total,count=fast_bin(subsets[k],image,nids)
            margin_sums[k]+=total
            margin_counts[k]+=count
    means=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0)
    margin_means=np.divide(margin_sums,margin_counts,
                           out=np.full_like(margin_sums,np.nan),where=margin_counts>0)
    return means,counts,margin_means,margin_counts


def micro_comparison(rows):
    d=pd.DataFrame(rows)
    result=[]
    for (marker,group,contrast,absdist), g in d.groupby(['marker','group','contrast','shift_distance_px']):
        if absdist==0:
            continue
        base=d[(d.marker==marker)&(d.group==group)&(d.contrast==contrast)&(d.shift_distance_px==0)]
        if len(base)!=1 or len(g)!=4:
            raise ValueError('Missing microshift(s)')
        med=g.median_delta.to_numpy(dtype=float)
        finite = med[np.isfinite(med)]
        result.append(dict(marker=marker,group=group,contrast=contrast,
                           shift_distance_px=int(absdist),observed_median_delta=float(base.iloc[0].median_delta),
                           n_controls=len(g),shift_median_min=float(np.min(finite)) if len(finite) else np.nan,
                           shift_median_median=float(np.median(finite)) if len(finite) else np.nan,
                           shift_median_max=float(np.max(finite)) if len(finite) else np.nan,
                           n_shift_medians_ge_observed=int(np.sum(med>=float(base.iloc[0].median_delta)))))
    return pd.DataFrame(result)


def margin_comparison(means,counts,ids,marker,group,lhs,rhs,label,min_pixels=10):
    """Compare deltas on common instance IDs for unablated and ablated regions."""
    rows=[]
    for i,margin in enumerate(MARGINS):
        eligible=(counts[i,lhs,ids]>=min_pixels)&(counts[i,rhs,ids]>=min_pixels)
        ids_eligible=ids[eligible]
        values=means[i,lhs,ids_eligible]-means[i,rhs,ids_eligible]
        rows.append({'marker':marker,'group':group,'contrast':label,
                     'excluded_boundary_distance_px':margin,'n_group':len(ids),
                     'n_paired':len(values),
                     'median_delta':float(np.median(values)) if len(values) else np.nan,
                     'fraction_positive':float(np.mean(values>0)) if len(values) else np.nan})
    return rows


def matched_margin_comparison(means,counts,ids,marker,group,lhs,rhs,label,min_pixels=10):
    """Controls dropout: compare original and margin-ablated effects on the SAME IDs."""
    r=[]
    for index in [1,2]:
        original=(counts[0,lhs,ids]>=min_pixels)&(counts[0,rhs,ids]>=min_pixels)
        ablated=(counts[index,lhs,ids]>=min_pixels)&(counts[index,rhs,ids]>=min_pixels)
        good=ids[original & ablated]
        v0=means[0,lhs,good]-means[0,rhs,good]
        v1=means[index,lhs,good]-means[index,rhs,good]
        r.append({'marker':marker,'group':group,'contrast':label,
                  'excluded_boundary_distance_px':MARGINS[index],
                  'n_group':len(ids),'n_common_paired':len(good),
                  'n_original_paired':int(original.sum()),
                  'n_ablated_paired':int(ablated.sum()),
                  'median_original_on_common':float(np.median(v0)) if len(good) else np.nan,
                  'median_ablated_on_common':float(np.median(v1)) if len(good) else np.nan,
                  'median_within_instance_change':float(np.median(v1-v0)) if len(good) else np.nan,
                  'fraction_change_positive':float(np.mean(v1>v0)) if len(good) else np.nan})
    return r


def ensure_11c_integrity(path):
    if not path.is_dir():raise FileNotFoundError(f'WF11C output not found: {path}')
    h=json.loads((path/'output_sha256.json').read_text())
    if {p.name for p in path.iterdir() if p.is_file()} != set(h)|{'output_sha256.json'}:
        raise RuntimeError('WF11C output manifest mismatch')
    for name,digest in h.items():
        if sha(path/name)!=digest:raise RuntimeError(f'WF11C changed: {name}')
    return {name:sha(path/name) for name in sorted(set(h)|{'output_sha256.json'})}


def hashes_for_outputs(stage):
    digest={p.name:sha(p) for p in sorted(stage.iterdir()) if p.is_file()}
    (stage/'output_sha256.json').write_text(json.dumps(digest,indent=2)+'\n')
    return len(digest)


def build(args):
    if args.out.exists():raise FileExistsError(f'Output exists: {args.out}')
    if args.min_pixels!=10:raise ValueError('min-pixels must remain 10 for WF11 matching')
    wf11_hashes=ensure_run_integrity(args.workflow11)
    wf11c_hashes=ensure_11c_integrity(args.workflow11c)
    summary,paths,base,_,input_hashes=file_manifest(args)
    markers=tuple(x.strip() for x in args.markers.split(','))
    if not markers or len(set(markers))!=len(markers) or set(markers)!=set(DEFAULT_MARKERS):
        raise ValueError('All five certified marker names required: '+','.join(DEFAULT_MARKERS))
    long=pd.read_csv(args.workflow11/'instance_marker_region_signal.csv')
    mask=tifffile.imread(args.mask)
    if tuple(mask.shape)!=tuple(summary['mask_shape_assumed_zyx']):raise ValueError('Mask shape mismatch')
    nids=int(mask.max())
    ids=base.instance_id.to_numpy(dtype=np.int64)
    if set(np.unique(mask))- {0} != set(ids):raise ValueError('Mask ID set mismatch')
    cache=[]
    for z in range(mask.shape[0]):
        ge=cached_geometry(mask[z],nids)
        cache.append(ge)
        print(f'GEOMETRY z={z:02d}: owner boundaries={ge[1]:,}, assigned={ge[2]:,}',flush=True)
    groups,thr=group_ids(base)
    micro_rows=[]; margin_rows=[]; margin_matched=[]
    for marker in markers:
        print('MARKER:',marker,flush=True)
        mm,cc,ab,ac=accumulate_marker(mask,paths[marker],nids,cache)
        check_zero_shift_matches_wf11(mm[0],cc[0],long,marker,ids,args.min_pixels)
        if not np.array_equal(cc[0],ac[0]) or not np.allclose(mm[0],ab[0],equal_nan=True):
            raise RuntimeError('Ablation margin=0 must match zero-shift baseline')
        for i,(dy,dx) in enumerate(MICRO_OFFSETS):
            for group,gr_ids in groups.items():
                for lhs,rhs,label in COMPARISONS:
                    r=summarize_contrast(mm[i],cc[i],gr_ids,args.min_pixels,
                                         lhs,rhs,marker,group,dy,dx,label)
                    micro_rows.append(r)
        for group,gr_ids in groups.items():
            for lhs,rhs,label in COMPARISONS:
                margin_rows.extend(margin_comparison(ab,ac,gr_ids,marker,group,
                                   lhs,rhs,label,args.min_pixels))
                margin_matched.extend(matched_margin_comparison(ab,ac,gr_ids,marker,group,
                                   lhs,rhs,label,args.min_pixels))
    dfmicro=pd.DataFrame(micro_rows)
    dfmicro_summary=micro_comparison(dfmicro)
    dfmargin=pd.DataFrame(margin_rows)
    dfmatched=pd.DataFrame(margin_matched)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.workflow11d_tmp_',dir=args.out.parent))
    try:
        dfmicro.to_csv(stage/'microshift_contrasts.csv',index=False)
        dfmicro_summary.to_csv(stage/'microshift_vs_unshifted.csv',index=False)
        dfmargin.to_csv(stage/'boundary_ablation_unmatched.csv',index=False)
        dfmatched.to_csv(stage/'boundary_ablation_common_instances.csv',index=False)
        record={
            'workflow':'11D Small XY microshift + Voronoi ownership-boundary ablation',
            'version':'0.3.0-exploratory',
            'markers':list(markers),'n_instances':len(ids),
            'microshifts_dy_dx':list(map(list,MICRO_OFFSETS)),
            'microshift_meaning':'translation of intensity map relative to fixed pseudo-labels, no wrap',
            'neighbor_competition_proxy':'Euclidean distance to 4-neighbor edge of different nearest-nucleus Voronoi ownership',
            'boundary_exclusions_pixels':list(MARGINS),
            'geometry_per_z':[{'z':i,'n_owner_boundary_pixels':g[1],
                              'n_assigned_pixels':g[2],
                              'n_kept_by_margin':g[3]} for i,g in enumerate(cache)],
            'min_finite_pixels':args.min_pixels,
            'n_micro_rows':len(dfmicro),
            'n_micro_summary_rows':len(dfmicro_summary),
            'n_ablation_rows':len(dfmargin),
            'n_ablation_matched_rows':len(dfmatched),
            'n_roi_high':len(groups['ROI_high_selected']),
            'her2_selected_from_unshifted_data_P99':float(thr),
            'input_sha256':{str(k):v for k,v in input_hashes.items()},'source_wf11_hashes':wf11_hashes,
            'source_wf11c_hashes':wf11c_hashes,
            'limitations':[
                'One specimen; no independent validation or inferential p-values.',
                'Nuclear-like watershed pseudo-instance != whole cell; no true membrane attribution.',
                'Boundary mask marks competition between nearest-nucleus Voronoi domains, NOT known cell-cell contact.',
                'Boundary exclusion is a proximity/ambiguity sensitivity test; cannot remove all other-cell marker contamination.',
                'Z is section index, physical spacing unknown. All distances are within-plane XY pixels.',
                'An XY shift of 1-4 pixels measures precision sensitivity, NOT a biological or proper random null.',
                'Marker-to-mask registration has not been independently certified.',
                'HER2-high selection and evaluation use one specimen; avoid confirmatory claims.',
                'Different marker raw intensity units not directly comparable.'
            ]
        }
        (stage/'robustness_summary.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
        nfiles=hashes_for_outputs(stage)
        stage.rename(args.out)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return {'status':'CREATED','n_instances':len(ids),'markers':len(markers),
            'shifts':len(MICRO_OFFSETS),'microshift_rows':len(dfmicro),
            'boundary_matched_rows':len(dfmatched),'result_files':nfiles,
            'zero_shift_reference':'PASS','output':str(args.out)}


def verify(args):
    if not args.out.is_dir():raise FileNotFoundError(args.out)
    h=json.loads((args.out/'output_sha256.json').read_text())
    if {p.name for p in args.out.iterdir() if p.is_file()} != set(h)|{'output_sha256.json'}:
        raise RuntimeError('Output file manifest mismatch')
    for p,d in h.items():
        if sha(args.out/p)!=d:raise RuntimeError(f'Output changed: {p}')
    s=json.loads((args.out/'robustness_summary.json').read_text())
    if ensure_run_integrity(args.workflow11)!=s['source_wf11_hashes']:
        raise RuntimeError('Source WF11 changed')
    if ensure_11c_integrity(args.workflow11c)!=s['source_wf11c_hashes']:
        raise RuntimeError('Source WF11C changed')
    _,_,base,_,fingerprints=file_manifest(args)
    if {str(k):v for k,v in fingerprints.items()}!=s['input_sha256'] or len(base)!=s['n_instances']:
        raise RuntimeError('Raw inputs or WF09 changed')
    for name,key in [('microshift_contrasts.csv','n_micro_rows'),
                     ('microshift_vs_unshifted.csv','n_micro_summary_rows'),
                     ('boundary_ablation_unmatched.csv','n_ablation_rows'),
                     ('boundary_ablation_common_instances.csv','n_ablation_matched_rows')]:
        if len(pd.read_csv(args.out/name))!=s[key]:raise RuntimeError('Output row count mismatch '+name)
    return {'status':'PASS','verified_results':len(h),'verified_inputs':len(fingerprints),
            'verified_wf11_reference':len(s['source_wf11_hashes']),
            'verified_wf11c_reference':len(s['source_wf11c_hashes']),
            'n_instances':s['n_instances']}


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--apply',action='store_true')
    mode.add_argument('--verify',action='store_true')
    p.add_argument('--mask',type=Path,default=Path('data/inspect/LVILymph/full_mask_final_segmentation_hwatershed_bg500_90%.tif'))
    p.add_argument('--manifest',type=Path,default=Path('configs/protein_marker_manifest_registered_5channels.json'))
    p.add_argument('--workflow09',type=Path,default=Path('runs/3d_protein_marker_integration_v1'))
    p.add_argument('--workflow11',type=Path,default=Path('runs/3d_spatial_signal_attribution_v0_1'))
    p.add_argument('--workflow11c',type=Path,default=Path('runs/3d_signal_attribution_shift_controls_v0_2'))
    p.add_argument('--out',type=Path,default=Path('runs/3d_signal_attribution_boundary_robustness_v0_3'))
    p.add_argument('--min-pixels',type=int,default=10)
    p.add_argument('--markers',type=str,default=','.join(DEFAULT_MARKERS))
    args=p.parse_args(argv)
    if args.check:
        h=ensure_run_integrity(args.workflow11)
        c=ensure_11c_integrity(args.workflow11c)
        summary,paths,df,_,source=file_manifest(args)
        if args.out.exists():
            raise FileExistsError('Refusing to reuse existing run folder: '+str(args.out))
        obj={'status':'PREFLIGHT_PASS','n_instances':len(df),
             'source_files':len(source),'wf11_reference_files':len(h),
             'wf11c_reference_files':len(c),'markers':list(paths),
             'n_microshifts':len(MICRO_OFFSETS),'n_boundary_margins':len(MARGINS)}
    elif args.verify:
        obj=verify(args)
    else:
        obj=build(args)
    print(json.dumps(obj,indent=2,ensure_ascii=False))


if __name__=='__main__':cli()
