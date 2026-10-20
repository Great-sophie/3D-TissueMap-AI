#!/usr/bin/env python3
"""Frozen external-volume Z-sampling benchmark for LVILymph aligned 3D IMC.
Uses author-provided full pseudo-instance mask, NOT validated single-cell ground truth.
Data-derived tuning is prohibited; baseline definitions are frozen before seeing results.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
from collections import defaultdict
from zipfile import ZipFile
import numpy as np
import tifffile

MARKERS = {
    'Gd156': 'c-erbB-2_HER2_provisional',
    'Cd112': 'panCytokeratin',
    'Eu151': 'CD31',
    'Yb174': 'E_Cadherin',
    'Er168': 'LYVE1',
    'Ir193': 'Nuclear_Ir193',
}
PRIMARY = ('Gd156', 'Cd112')
OTHER = ('Eu151', 'Yb174', 'Er168')
METHODS = ('nearest1', 'nearest2', 'all_others')

def write_csv(path, data, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(data)

def bootstrap_block(deltas, blocks, seed=20261019, n_boot=999):
    # Cluster bootstrap: sample XY/Z spatial blocks, keep all objects in block.
    sums, sizes = defaultdict(float), defaultdict(int)
    for d, key in zip(deltas, blocks):
        sums[key] += float(d)
        sizes[key] += 1
    keys = sorted(sums)
    if len(keys) < 2:
        return float('nan'), float('nan'), len(keys)
    arr_sum = np.array([sums[k] for k in keys], dtype=float)
    arr_n = np.array([sizes[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        choice = rng.integers(0, len(keys), size=len(keys))
        bs[b] = arr_sum[choice].sum() / arr_n[choice].sum()
    lo, hi = np.quantile(bs, [.025, .975])
    return float(lo), float(hi), len(keys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/inspect/LVILymph')
    ap.add_argument('--out', default='runs/lvilymph_external_sampling_v0_1')
    ap.add_argument('--z-step-um', type=float, default=2.0, help='Assumed from original protocol; verify if using physical conclusions')
    ap.add_argument('--xy-step-um', type=float, default=1.0, help='Assumed native IMC pixel size; verify from provenance')
    ap.add_argument('--block-xy-um', type=float, default=32.0)
    ap.add_argument('--block-z-um', type=float, default=24.0)
    ap.add_argument('--min-z', type=int, default=4, help='Fixed mask-only eligibility criterion')
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite: {out}')
    src = root / 'LVILymphBreastCancer_SIMILARITY10_In115.zip'
    mask_file = root / 'full_mask_final_segmentation_hwatershed_bg500_90%.tif'
    if not src.is_file() or not mask_file.is_file():
        raise FileNotFoundError(f'Expected aligned archive {src} and full mask {mask_file}')
    print('[1/5] Load fixed source full mask and verify aligned TIFF membership', flush=True)
    mask = tifffile.imread(mask_file)
    if mask.ndim != 3 or mask.shape[0] != 16:
        raise ValueError(f'Expected 16ZYX full mask, got {mask.shape}')
    zdim, h, w = mask.shape
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f'Mask needs integer labels, got {mask.dtype}')
    # Only record same-label Z presence and mask-object geometry; no resegmentation.
    geometry = defaultdict(dict)
    per_z_count = []
    label_sets = []
    for z in range(zdim):
        m = mask[z]
        yy, xx = np.nonzero(m)
        ids, inv, counts = np.unique(m[yy, xx], return_inverse=True, return_counts=True)
        sumx = np.bincount(inv, weights=xx, minlength=len(ids))
        sumy = np.bincount(inv, weights=yy, minlength=len(ids))
        # XY edge flag based on full registered field of view.
        touching = np.unique(np.concatenate([m[0], m[-1], m[:,0], m[:,-1]]))
        touching = set(map(int, touching[touching > 0]))
        for j, ident in enumerate(ids):
            geometry[int(ident)][z] = (int(counts[j]), float(sumx[j]/counts[j]), float(sumy[j]/counts[j]), int(ident) in touching)
        per_z_count.append({'section': z+1, 'n_pseudo_ids': int(len(ids)), 'n_positive_voxels': int(len(xx))})
        label_sets.append(set(map(int, ids)))
    multi = [ident for ident, d in geometry.items() if len(d) >= 2]
    gapped = [ident for ident, d in geometry.items() if max(d)-min(d)+1 != len(d)]
    same_xy = set()
    face_contacts = defaultdict(set)
    for z in range(zdim-1):
        a,b = mask[z], mask[z+1]
        same = a[(a == b) & (a > 0)]
        if len(same):
            for v in np.unique(same):
                same_xy.add(int(v))
                face_contacts[int(v)].add(z)
    eligible = {}
    for ident, d in geometry.items():
        sections = sorted(d)
        if len(sections) < args.min_z or max(sections)-min(sections)+1 != len(sections):
            continue
        if any(v[3] for v in d.values()):
            continue
        # Conservative face-to-face Z continuity; does not validate within-plane connectedness.
        if not all(z in face_contacts[ident] for z in sections[:-1]):
            continue
        target = sections[(len(sections)-1)//2]
        eligible[ident] = target
    print(f'   source IDs={len(geometry)}, multi-Z={len(multi)}, gap-ID={len(gapped)}, '
          f'positive XY-connected across consecutive Z={len(same_xy)}, eligible={len(eligible)}', flush=True)

    with ZipFile(src) as zipf:
        lookup = defaultdict(dict)
        for name in zipf.namelist():
            if not name.lower().endswith(('.tif', '.tiff')):
                continue
            p = Path(name)
            if p.parent.name not in MARKERS:
                continue
            try: zindex = int(p.name.split('_',1)[0]) - 1
            except ValueError: continue
            if zindex in lookup[p.parent.name]:
                raise ValueError(f'Duplicate section: {p.parent.name} {zindex+1}')
            lookup[p.parent.name][zindex] = name
        for metal in MARKERS:
            if set(lookup[metal]) != set(range(zdim)):
                raise ValueError(f'{metal} missing aligned planes: {sorted(set(range(zdim))-set(lookup[metal]))}')
        print('[2/5] Quantify object signals for preselected 6 markers (16 aligned planes)', flush=True)
        # Only per-z means enter predictions; avoid full 16Z x 29C tensor.
        by_object = defaultdict(dict)
        rows = []
        fg_bg_qc = []
        for z in range(zdim):
            m = mask[z]
            yy, xx = np.nonzero(m)
            ids, inv, counts = np.unique(m[yy, xx], return_inverse=True, return_counts=True)
            means = {}
            for metal in MARKERS:
                plane = tifffile.imread(io.BytesIO(zipf.read(lookup[metal][z])))
                if plane.shape != (h,w):
                    raise ValueError(f'{metal} z={z+1}: image {plane.shape} != full mask {(h,w)}')
                sig = plane[yy,xx].astype('float64', copy=False)
                sums = np.bincount(inv, weights=sig, minlength=len(ids))
                means[metal] = sums/counts
                if z in (0,zdim//2,zdim-1):
                    bg = plane[mask[z] == 0]
                    fg_bg_qc.append({'z_section':z+1,'isotope':metal,
                        'foreground_mean':float(sig.mean()) if len(sig) else np.nan,
                        'background_mean':float(bg.mean()) if len(bg) else np.nan,
                        'foreground_positive_frac':float(np.mean(sig>0)) if len(sig) else np.nan})
            for j, ident in enumerate(ids):
                k = int(ident)
                item = {'pseudo_id':k,'z_section':z+1,'n_voxels':int(counts[j]),
                        'centroid_x_px':geometry[k][z][1],'centroid_y_px':geometry[k][z][2]}
                item.update({f'{metal}_mean': float(means[metal][j]) for metal in MARKERS})
                by_object[k][z] = item
                rows.append(item)
            print(f'   section {z+1:02d}/{zdim}: {len(ids)} pseudo-IDs',flush=True)

    print('[3/5] Fixed single middle-Z holdout: nearest1 / nearest2 / all-other',flush=True)
    evals = []
    for ident,target in eligible.items():
        d = by_object[ident]
        other = sorted((z for z in d if z != target), key=lambda z: (abs(z-target), z))
        if len(other) < 3: continue
        c = d[target]
        block = (int(np.floor(c['centroid_x_px']*args.xy_step_um/args.block_xy_um)),
                 int(np.floor(c['centroid_y_px']*args.xy_step_um/args.block_xy_um)),
                 int(np.floor(target*args.z_step_um/args.block_z_um)))
        for metal in (*PRIMARY,*OTHER):
            truth = c[f'{metal}_mean']
            if not np.isfinite(truth) or truth<0: continue
            predictions = {}
            for meth, zs in [('nearest1',other[:1]),('nearest2',other[:2]),('all_others',other)]:
                weights = np.array([d[z]['n_voxels'] for z in zs], dtype=float)
                v = np.array([d[z][f'{metal}_mean'] for z in zs],dtype=float)
                predictions[meth] = float(np.average(v,weights=weights))
            item = {'pseudo_id':ident,'isotope':metal,'target_z_section':target+1,
                    'n_source_sections':len(d),'centroid_x_px':c['centroid_x_px'],
                    'centroid_y_px':c['centroid_y_px'],'block':':'.join(map(str,block)),
                    'target_mean':truth}
            for meth in METHODS:
                item[f'{meth}_prediction'] = predictions[meth]
                item[f'{meth}_error']=float(abs(np.log1p(predictions[meth])-np.log1p(truth)))
            evals.append(item)
    if not evals:
        raise RuntimeError('No eligible object-level targets. Source pseudo IDs may lack cross-Z continuity; do not force a 3D benchmark.')
    print('[4/5] Compute fixed spatial-cluster bootstrap for paired errors',flush=True)
    summaries, pairs = [], []
    for metal in (*PRIMARY,*OTHER):
        subset = [r for r in evals if r['isotope']==metal]
        if not subset: continue
        result = {'isotope':metal,'marker_label':MARKERS[metal],'n_objects':len(subset),
                  'n_spatial_blocks':len({r['block'] for r in subset})}
        for meth in METHODS:
            vals = np.array([r[f'{meth}_error'] for r in subset])
            result[f'{meth}_median_abs_log_error']=float(np.median(vals))
            result[f'{meth}_mean_abs_log_error']=float(np.mean(vals))
        summaries.append(result)
        for a,b in (('nearest2','nearest1'),('nearest2','all_others')):
            # Positive means a beats b: error_b - error_a
            deltas = np.array([r[f'{b}_error']-r[f'{a}_error'] for r in subset])
            lo,hi,nb = bootstrap_block(deltas,[r['block'] for r in subset])
            pairs.append({'isotope':metal,'method_a':a,'method_b':b,
                          'n_objects':len(subset),'n_blocks':nb,
                          'mean_error_improvement_a_vs_b':float(np.mean(deltas)),
                          'boot_p025':lo,'boot_p975':hi,
                          'fraction_objects_a_lower_error':float(np.mean(deltas>0))})
    # All checks are successful: create a new output directory just once, never overwrite.
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'per_z_object_signals.csv',rows,list(rows[0]))
    write_csv(out/'per_z_mask_qc.csv',per_z_count,list(per_z_count[0]))
    write_csv(out/'foreground_signal_qc.csv',fg_bg_qc,list(fg_bg_qc[0]))
    write_csv(out/'heldout_object_predictions.csv',evals,list(evals[0]))
    write_csv(out/'external_sampling_summary.csv',summaries,list(summaries[0]))
    write_csv(out/'paired_spatial_block_intervals.csv',pairs,list(pairs[0]))
    audit={'status':'complete','n_z':zdim,'n_aligned_channels_total':29,'n_quantified_channels':len(MARKERS),
        'mask_shape':list(map(int,mask.shape)), 'unique_source_pseudo_ids':len(geometry),
        'multi_z_source_ids':len(multi),'gapped_source_ids':len(gapped),
        'source_ids_with_any_same_xy_adjacent_z':len(same_xy),
        'n_eligible_mask_only_ids':len(eligible),'assumed_pixel_size_um':args.xy_step_um,
        'assumed_section_spacing_um':args.z_step_um,
        'eligible_criteria':f'>={args.min_z} contiguous sections, face-to-face adjacent-Z contact along full section span, no full-FOV XY edge contact, fixed middle-Z held out',
        'nearest1_tie_break':'choose smaller section index',
        'nearest2':'two closest available non-target sections, voxel-count-weighted mean',
        'all_others':'all available non-target sections, voxel-count-weighted mean',
        'reference':'same pseudo-object mean signal on withheld middle-Z plane; NOT biological ground truth',
        'markers':MARKERS, 'intensity_transform':'absolute log1p prediction error',
        'spatial_block_um':[args.block_xy_um,args.block_xy_um,args.block_z_um],
        'n_bootstrap':999,'random_seed':20261019,
        'source':'Author-supplied LVILymph full 3D pseudo-instance mask and channel-wise SIMILARITY10 registered TIFFs',
        'limitations':['The 16-section mask is not validated manual cell annotation',
                       'Different biological lesion / panel from SecondHer2; compare effect directions, not raw errors',
                       'This analysis uses no retraining, excludes target plane from predictors',
                       'Some marker names truncated in OME; Gd156 HER2 is provisional until matched to original panel',
                       'The full mask may contain disconnected fragments sharing an ID; face contacts alone are not complete 3D connected-component QC',
                       'Spatial bootstrap quantifies within-volume uncertainty, not between-patient generalization']}
    (out/'protocol_and_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    print('[5/5] Completed; results:',out,flush=True)
    print('\n=== 3D MASK ID AUDIT ===')
    print(f'  Unique IDs {len(geometry)} | multi-Z IDs {len(multi)} | gapped IDs {len(gapped)} | any 6-connected adjacent-Z IDs {len(same_xy)} | eligible {len(eligible)}')
    print('\n=== EXTERNAL Z-SAMPLING MEDIAN ABS LOG ERROR ===')
    for r in summaries:
        print(f"  {r['isotope']:5s} n={r['n_objects']:5d} blocks={r['n_spatial_blocks']:4d} " +
              ' '.join(f"{m}={r[f'{m}_median_abs_log_error']:.5f}" for m in METHODS))
    print('\n=== PAIRED BLOCK BOOTSTRAP (POSITIVE MEANS METHOD A BETTER) ===')
    for r in pairs:
        print(f"  {r['isotope']:5s} {r['method_a']} vs {r['method_b']}: "
              f"delta={r['mean_error_improvement_a_vs_b']:+.5f} "
              f"95% [{r['boot_p025']:+.5f}, {r['boot_p975']:+.5f}]")
    print('\nSUCCESS:',out.resolve())

if __name__=='__main__':
    main()
