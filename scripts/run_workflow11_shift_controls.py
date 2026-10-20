#!/usr/bin/env python3
"""WF11C v0.2 exploratory XY image-shift controls against *unchanged* pseudo-instance shells.

Only works on an existing verified WF11 v0.1 and WF09 run. No biological
cell-membrane attribution or hypothesis-test p-values are inferred.
"""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_signal_attribution_core import REGIONS, assign_slice
from run_spatial_signal_attribution import file_manifest, sha

OFFSETS = ((0, 0), (0, 16), (0, -16), (16, 0), (-16, 0),
           (0, 32), (0, -32), (32, 0), (-32, 0))
DEFAULT_MARKERS = ('HER2', 'panCK', 'CD31', 'LYVE1', 'Ir193')
COMPARISONS = ((1, 0, 'perinuclear_1_2px_minus_nuclear'),
               (1, 2, 'perinuclear_1_2px_minus_3_5px'))


def translated_no_wrap(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate intensity, not masks. Missing edges become NaN, never wrapped."""
    a = np.asarray(image)
    if a.ndim != 2:
        raise ValueError('2D marker image required')
    h, w = a.shape
    out = np.full((h, w), np.nan, dtype=np.float64)
    if abs(dy) >= h or abs(dx) >= w:
        return out
    y_out = slice(max(dy, 0), min(h + dy, h))
    x_out = slice(max(dx, 0), min(w + dx, w))
    y_in = slice(max(-dy, 0), min(h - dy, h))
    x_in = slice(max(-dx, 0), min(w - dx, w))
    out[y_out, x_out] = a[y_in, x_in]
    return out


def fast_bin(ownership, image, nids: int):
    """Return region x instance sums/counts excluding all NaN/inf."""
    selected, base_codes = ownership if isinstance(ownership, tuple) else (ownership.selected, ownership.codes)
    v = image.ravel()[selected]
    finite = np.isfinite(v)
    codes = base_codes[finite]
    totals = np.bincount(codes, weights=v[finite],
                         minlength=(nids + 1) * len(REGIONS))
    counts = np.bincount(codes, minlength=(nids + 1) * len(REGIONS))
    shape = len(REGIONS), nids + 1
    return totals.reshape(shape), counts.reshape(shape)


def means_for_marker_all_offsets(mask, marker_paths, nids, ownership_cache=None):
    """One TIFF read per plane; reuse nearest pseudo-ID geometry across offsets.

    An optional cached list of (selected_indices, dense_codes) avoids recomputing
    the 2D distance transform for the other marker channels.
    """
    n_offsets=len(OFFSETS)
    shape=(n_offsets,len(REGIONS),nids+1)
    sums=np.zeros(shape,np.float64)
    counts=np.zeros(shape,np.int64)
    if ownership_cache is not None and len(ownership_cache)!=mask.shape[0]:
        raise ValueError('Invalid cached ownership length')
    for z in range(mask.shape[0]):
        own=ownership_cache[z] if ownership_cache is not None else assign_slice(mask[z],nids)
        a=tifffile.imread(marker_paths[z])
        if a.shape!=mask.shape[1:]:
            raise ValueError(f'marker plane {z} shape mismatch: {a.shape}')
        for i,offset in enumerate(OFFSETS):
            t,c=fast_bin(own,translated_no_wrap(a,*offset),nids)
            sums[i]+=t
            counts[i]+=c
    means=np.divide(sums,counts,where=counts>0,out=np.full_like(sums,np.nan))
    return means,counts


def contrast_values(means, counts, ids, min_pixels, lhs, rhs):
    lhs_ok = counts[lhs, ids] >= min_pixels
    rhs_ok = counts[rhs, ids] >= min_pixels
    good = lhs_ok & rhs_ok
    ids_valid = ids[good]
    delta = means[lhs, ids_valid] - means[rhs, ids_valid]
    return ids_valid, delta


def summarize_contrast(means, counts, ids, min_pixels, lhs, rhs, marker,
                       group, dy, dx, name):
    idx, delta = contrast_values(means, counts, ids, min_pixels, lhs, rhs)
    return {'marker': marker, 'group': group, 'contrast': name,
            'dy': int(dy), 'dx': int(dx), 'shift_distance_px': float(np.hypot(dy, dx)),
            'n_group': len(ids), 'n_paired': len(idx),
            'fraction_paired': float(len(idx)/len(ids)) if len(ids) else np.nan,
            'median_delta': float(np.median(delta)) if len(delta) else np.nan,
            'fraction_positive': float(np.mean(delta > 0)) if len(delta) else np.nan}


def group_ids(base):
    if not base.instance_id.is_unique:
        raise ValueError('Nonunique instance ID')
    thr = float(base.mean_raw_HER2.quantile(.99))
    inroi = base.centroid_x_px.ge(480) & base.centroid_x_px.lt(720) & \
            base.centroid_y_px.ge(480) & base.centroid_y_px.lt(720)
    high = base.mean_raw_HER2.ge(thr)
    return {
        'ROI_all':base.loc[inroi,'instance_id'].to_numpy(dtype=np.int64),
        'ROI_high_selected':base.loc[inroi & high,'instance_id'].to_numpy(dtype=np.int64),
        'ROI_other':base.loc[inroi & ~high,'instance_id'].to_numpy(dtype=np.int64),
        'outside_ROI':base.loc[~inroi,'instance_id'].to_numpy(dtype=np.int64),
    }, thr


def ensure_run_integrity(root):
    """Validate last frozen WF11 artifacts before comparing to shift=0."""
    out_hash_file = root / 'output_sha256.json'
    hashes = json.loads(out_hash_file.read_text())
    available = {p.name for p in root.iterdir() if p.is_file()}
    if not set(hashes).issubset(available):
        raise RuntimeError('A manifest-listed WF11 result is missing')
    # Users may have appended downstream exploratory CSVs (e.g. paired_signal_validation.csv).
    # These are not part of the original immutable WF11 output manifest and are ignored.
    for name, digest in hashes.items():
        if sha(root / name) != digest:
            raise RuntimeError(f'WF11 output hash changed: {name}')
    return {name:sha(root / name) for name in sorted(set(hashes)|{out_hash_file.name})}


def check_zero_shift_matches_wf11(means, counts, long, marker, ids, min_pixels=10):
    g = long[long.marker.eq(marker)]
    if len(g) != len(ids) * len(REGIONS):
        raise ValueError('Unexpected WF11 marker rows')
    idx = {str(x): i for i,x in enumerate(REGIONS)}
    for region, r in idx.items():
        part = g[g.region.eq(region)].set_index('instance_id').reindex(ids)
        if part.index.has_duplicates or part.mean_raw_signal.isna().all() and r == 0:
            raise ValueError('WF11 reference region invalid')
        expected_count = part.n_finite_signal_voxels.to_numpy(dtype=int)
        if not np.array_equal(expected_count, counts[r, ids]):
            raise RuntimeError(f'zero-shift counts mismatch: {marker} / {region}')
        expected = part.mean_raw_signal.to_numpy(dtype=float)
        got = means[r, ids].copy()
        got[counts[r, ids] < min_pixels] = np.nan
        if not np.allclose(got, expected, rtol=2e-7, atol=2e-7, equal_nan=True):
            errors = np.abs(got - expected)
            raise RuntimeError(f'zero-shift mean mismatch: {marker} / {region}: '+str(np.nanmax(errors)))


def digest_results(stage):
    files = {p.name:sha(p) for p in sorted(stage.iterdir()) if p.is_file()}
    (stage/'output_sha256.json').write_text(json.dumps(files,indent=2)+'\n')
    return len(files)


def build_comparison(rows):
    d=pd.DataFrame(rows)
    r=[]
    for (marker, group, contrast), g in d.groupby(['marker','group','contrast'],sort=True):
        original=g[(g.dy==0) & (g.dx==0)]
        null=g[(g.dy!=0) | (g.dx!=0)]
        if len(original)!=1 or len(null)!=len(OFFSETS)-1:
            raise ValueError('Shift grid incomplete')
        actual=original.iloc[0]
        v=null.median_delta.to_numpy(dtype=float)
        p=null.fraction_positive.to_numpy(dtype=float)
        r.append({'marker':marker,'group':group,'contrast':contrast,
                  'observed_median_delta':float(actual.median_delta),
                  'observed_fraction_positive':float(actual.fraction_positive),
                  'observed_n_paired':int(actual.n_paired),
                  'shifted_median_min':float(np.nanmin(v)),
                  'shifted_median_median':float(np.nanmedian(v)),
                  'shifted_median_max':float(np.nanmax(v)),
                  'shifted_fraction_positive_median':float(np.nanmedian(p)),
                  'n_controls':len(null),
                  'n_null_with_median_ge_observed':int(np.sum(v>=actual.median_delta))})
    return pd.DataFrame(r)


def run(args):
    if args.out.exists():
        raise FileExistsError('Output already exists: '+str(args.out))
    wf11_hashes=ensure_run_integrity(args.workflow11)
    summary, marker_paths, base, _, input_hashes=file_manifest(args)
    markers=tuple(args.markers.split(','))
    if not markers or len(set(markers))!=len(markers) or not set(markers).issubset(set(DEFAULT_MARKERS)):
        raise ValueError('Expected comma separated names drawn from '+','.join(DEFAULT_MARKERS))
    if args.min_pixels!=10:
        raise ValueError('Keep min-pixels=10 to reproduce verified WF11 exactly')
    ref=pd.read_csv(args.workflow11/'instance_marker_region_signal.csv')
    groups,threshold=group_ids(base)
    mask=tifffile.imread(args.mask)
    if mask.ndim!=3 or tuple(mask.shape)!=tuple(summary['mask_shape_assumed_zyx']):
        raise ValueError('Mask shape unexpected')
    ids=base.instance_id.to_numpy(dtype=np.int64)
    nids=int(mask.max())
    if set(np.unique(mask)) - {0} != set(ids):
        raise ValueError('Mask IDs and reference differ')
    # 16 XY Voronoi assignments are reused across five marker channels and nine shifts.
    # Cache only integer indices / codes (not the large distance maps).
    ownership_cache=[]
    for z in range(mask.shape[0]):
        own=assign_slice(mask[z],nids)
        ownership_cache.append((own.selected,own.codes))
    rows=[]
    for marker in markers:
        print(f'MARKER: {marker}',flush=True)
        all_means,all_counts=means_for_marker_all_offsets(mask,marker_paths[marker],nids,ownership_cache)
        for i,(dy,dx) in enumerate(OFFSETS):
            means,counts=all_means[i],all_counts[i]
            if i==0:
                check_zero_shift_matches_wf11(means,counts,ref,marker,ids,args.min_pixels)
                print('  ZERO-SHIFT BASELINE: PASS',flush=True)
            for group,grouped_ids in groups.items():
                for lhs,rhs,label in COMPARISONS:
                    rows.append(summarize_contrast(means,counts,grouped_ids,args.min_pixels,
                                                   lhs,rhs,marker,group,dy,dx,label))
            print(f'  SHIFT dy={dy:+d} dx={dx:+d}: PASS',flush=True)
    raw=pd.DataFrame(rows)
    comp=build_comparison(rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.workflow11c_tmp_',dir=args.out.parent))
    try:
        raw.to_csv(stage/'shift_control_contrasts.csv',index=False)
        comp.to_csv(stage/'observed_vs_shift_controls.csv',index=False)
        report={
            'workflow':'11C-SameZ XY marker image shift negative controls',
            'version':'0.2.0-exploratory',
            'offset_convention':'dy/dx translates raw marker image relative to unchanged nuclear mask; out-of-field=NaN, no wrapping',
            'offsets_dy_dx':list(map(list,OFFSETS)),
            'markers':list(markers),'n_instances':len(base),
            'roi':{'x':[480,720],'y':[480,720]},
            'her2_high_selection':'same-specimen WF09 nuclear HER2 >= global P99; exploratory, circular selection for HER2',
            'her2_high_threshold':threshold,'n_roi_high':len(groups['ROI_high_selected']),
            'min_support_finite_pixels':args.min_pixels,
            'fixed_mask_sha256':sha(args.mask),
            'input_sha256':input_hashes,
            'source_wf11_hashes':wf11_hashes,
            'n_result_rows':len(raw),
            'limitations':[
                'Shifting a signal map relative to fixed pseudo-instance masks is a sensitivity control, not an independently sampled biological null.',
                'Nearby shifts can retain tissue-scale spatial autocorrelation; distant shifts may leave a different tissue compartment.',
                'HER2-high was selected by the unshifted HER2 signal. A shifted HER2 comparison of this group is selection biased and cannot be assigned a significance level.',
                'Rings are XY pixel distances from nuclear-like pseudo-label masks, not measured cell membrane or true cell territory.',
                'Ir193 itself helped form labels; its gradient is not an independent check of biology.',
                'All results from one specimen, independently unverified registration; no p-values or clinical claims.'
            ]}
        (stage/'control_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        digest_results(stage)
        stage.rename(args.out)
    finally:
        if stage.exists():shutil.rmtree(stage)
    return {'status':'CREATED','out':str(args.out),'markers':len(markers),
            'offsets':len(OFFSETS),'rows':len(raw),'zero_shift_baseline':'PASS'}


def verify(args):
    if not args.out.is_dir():raise FileNotFoundError('Shift-control results not found')
    h=json.loads((args.out/'output_sha256.json').read_text())
    if {p.name for p in args.out.iterdir() if p.is_file()} != set(h)|{'output_sha256.json'}:
        raise RuntimeError('Output manifest mismatch')
    for name,digest in h.items():
        if sha(args.out/name)!=digest:raise RuntimeError('Changed output: '+name)
    record=json.loads((args.out/'control_summary.json').read_text())
    if record['fixed_mask_sha256']!=sha(args.mask):
        raise RuntimeError('Mask changed')
    if record['source_wf11_hashes']!=ensure_run_integrity(args.workflow11):
        raise RuntimeError('WF11 reference has changed')
    _,_,_,_,current=file_manifest(args)
    if current != record['input_sha256']:
        raise RuntimeError('Source channels changed')
    count=len(pd.read_csv(args.out/'shift_control_contrasts.csv'))
    if count != record['n_result_rows']:
        raise RuntimeError('Result row count incorrect')
    return {'status':'PASS','verified_output_files':len(h),
            'verified_WF11_reference_files':len(record['source_wf11_hashes']),
            'verified_input_files':len(current),'contrast_rows':count}


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    runmode=p.add_mutually_exclusive_group(required=True)
    runmode.add_argument('--check',action='store_true')
    runmode.add_argument('--apply',action='store_true')
    runmode.add_argument('--verify',action='store_true')
    p.add_argument('--mask',type=Path,default=Path('data/inspect/LVILymph/full_mask_final_segmentation_hwatershed_bg500_90%.tif'))
    p.add_argument('--manifest',type=Path,default=Path('configs/protein_marker_manifest_registered_5channels.json'))
    p.add_argument('--workflow09',type=Path,default=Path('runs/3d_protein_marker_integration_v1'))
    p.add_argument('--workflow11',type=Path,default=Path('runs/3d_spatial_signal_attribution_v0_1'))
    p.add_argument('--out',type=Path,default=Path('runs/3d_signal_attribution_shift_controls_v0_2'))
    p.add_argument('--min-pixels',type=int,default=10)
    p.add_argument('--markers',type=str,default=','.join(DEFAULT_MARKERS))
    args=p.parse_args(argv)
    if args.check:
        h=ensure_run_integrity(args.workflow11)
        s,files,base,_,fingerprints=file_manifest(args)
        result={'status':'PREFLIGHT_PASS','wf11_reference_files':len(h),
                'input_files':len(fingerprints),'markers':list(files),
                'instances':len(base),'offsets':list(map(list,OFFSETS)),
                'output_exists':args.out.exists()}
    elif args.verify:
        result=verify(args)
    else:
        result=run(args)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':cli()
