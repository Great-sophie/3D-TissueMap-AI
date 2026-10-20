#!/usr/bin/env python3
"""Read-only diagnostic: where does 3D instance reconstruction fail?

Compares one trained V2 head checkpoint with reference-derived (ORACLE) seeds.
Oracle runs use watershed pseudo-label IDs and are NOT deployable algorithms or
independent validation. Only the existing VAL tiles are accessed; no training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from train_segmentation import read_data
from train_instance_heads_v2 import make_model, predict_heads, guided_watershed
from instance_watershed_baseline import compare_instances
from optimize_instance_separation import watershed_instances


def oracle_markers(ref: np.ndarray, allowed: np.ndarray):
    """One reference-ID seed, constrained to an allowed mask; for diagnosis only."""
    if ref.shape != allowed.shape or ref.ndim != 3:
        raise ValueError('Expected matching 3D reference and mask')
    ref = np.asarray(ref)
    allowed = np.asarray(allowed, bool)
    markers = np.zeros(ref.shape, dtype=np.int32)
    boxes = ndi.find_objects(ref.astype(np.int32, copy=False))
    ids = np.unique(ref)
    ids = ids[ids > 0]
    n = 0
    missing = 0
    for id_ in ids:
        id_ = int(id_)
        sl = boxes[id_ - 1] if id_ - 1 < len(boxes) else None
        if sl is None:
            missing += 1
            continue
        is_id = ref[sl] == id_
        candidates = is_id & allowed[sl]
        if not candidates.any():
            missing += 1
            continue
        # Center stays inside a valid reference instance, not the bounding box.
        dist = ndi.distance_transform_edt(is_id).astype(np.float32)
        dist[~candidates] = -1
        local = np.unravel_index(int(np.argmax(dist)), dist.shape)
        global_idx = tuple(s.start + v for s, v in zip(sl, local))
        n += 1
        markers[global_idx] = n
    return markers, {'n_reference_ids': int(len(ids)), 'n_oracle_seeds': n,
                     'n_reference_ids_missing_predicted_foreground': missing}


def decode_with_oracle(ref: np.ndarray, fg: np.ndarray, learned_boundary=None):
    """Reference-derived seeds + fixed watershed elevation: a diagnostic ceiling."""
    markers, stats = oracle_markers(ref, fg)
    if not np.any(markers):
        return np.zeros(ref.shape, dtype=np.int32), stats
    dist = ndi.distance_transform_edt(fg).astype(np.float32)
    elevation = -ndi.gaussian_filter(dist, 0.8)
    if learned_boundary is not None:
        elevation = elevation + 2.0 * np.asarray(learned_boundary, dtype=np.float32)
    pred = watershed(elevation, markers, mask=fg).astype(np.int32)
    return pred, stats


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size < 2 or a.std() < 1e-8 or b.std() < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    ckpt = args.head_run_dir / 'instance_heads_v2_3d.pt'
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    ck = torch.load(ckpt, map_location='cpu', weights_only=True)
    if ck.get('output_heads') != ['foreground', 'xy_inter_instance_interface', 'edt_interior_center']:
        raise ValueError('Unexpected V2 checkpoint head order')
    model = make_model(int(ck['base']), None)
    model.load_state_dict(ck['state_dict'], strict=True)
    model.to(device).eval()
    img, ref, _, _ = read_data(args.data_dir)
    split = pd.read_csv(args.split_run_dir / 'tile_splits.csv')
    val = split.loc[split.split == 'val'].copy()
    if val.empty:
        raise ValueError('No validation tiles')
    out_rows = []
    for i, row in enumerate(val.itertuples(index=False), 1):
        y, x, w = int(row.y), int(row.x), int(row.w)
        with torch.inference_mode():
            probs = predict_heads(model, img[:, y:y+w, x:x+w], device)
        g = args.guard
        if w <= 2*g:
            raise ValueError('Guard greater than tile size')
        sl = (slice(None), slice(g, w-g), slice(g, w-g))
        fg_prob, boundary_prob, center_prob = [np.asarray(p[sl]) for p in probs]
        rr = ref[:, y+g:y+w-g, x+g:x+w-g]
        fg = fg_prob >= 0.5
        # SAME model foreground for each non-oracle baseline.
        reference_ids = np.unique(rr)
        reference_ids = reference_ids[reference_ids > 0]
        experiments = {}
        base, _ = watershed_instances(fg_prob, min_distance=2, sigma=.8,
                                      threshold=.5, ensure_each_component=True)
        experiments['distance_same_model'] = (base, None)
        guided, gstats = guided_watershed(fg_prob, boundary_prob, center_prob,
                                          fg_threshold=.5, center_threshold=.35,
                                          min_distance=3, boundary_strength=2.0)
        experiments['learned_center_boundary'] = (guided, gstats)
        oracle_ref, oref_stats = decode_with_oracle(rr, rr > 0)
        experiments['ORACLE_reference_fg_oracle_seeds'] = (oracle_ref, oref_stats)
        oracle_pred, opred_stats = decode_with_oracle(rr, fg)
        experiments['ORACLE_pred_fg_oracle_seeds'] = (oracle_pred, opred_stats)
        oracle_pred_b, _ = decode_with_oracle(rr, fg, boundary_prob)
        experiments['ORACLE_pred_fg_oracle_seeds_learned_boundary'] = (oracle_pred_b, opred_stats)
        for name, (pred, meta) in experiments.items():
            scores = compare_instances(rr, pred, iou_threshold=.5)
            out_rows.append({'tile':i, 'y':y, 'x':x, 'method':name,
                             **scores, 'n_seeds': None if meta is None else meta.get('n_seeds', meta.get('n_oracle_seeds')),
                             'n_missing_ref_seeds': None if meta is None else meta.get('n_reference_ids_missing_predicted_foreground')})
        mask = fg
        print(f'VAL {i}/{len(val)}: reference_ids={len(reference_ids)}; '
              f'center_q90={np.quantile(center_prob[mask],.9) if mask.any() else float("nan"):.3f}; '
              f'center/boundary corr={safe_corr(center_prob[mask],boundary_prob[mask])}', flush=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(args.out / 'oracle_decoder_per_tile.csv', index=False)
    ranking = []
    for name, group in df.groupby('method'):
        ranking.append({'method':name, 'median_pq':float(group.pq.median()),
                        'median_recall':float(group.recall.median()),
                        'median_precision':float(group.precision.median()),
                        'median_predicted':float(group.n_predicted.median()),
                        'median_seed_count': None if group.n_seeds.isna().all() else float(group.n_seeds.median())})
    ranking.sort(key=lambda r:r['median_pq'], reverse=True)
    hist_path = args.head_run_dir / 'train_history.csv'
    history = None
    if hist_path.exists():
        h = pd.read_csv(hist_path)
        history = {'steps':int(len(h))}
        for col in ['n_center_target', 'n_boundary_target']:
            if col in h:
                history[col+'_zero_fraction'] = float((h[col] == 0).mean())
                history[col+'_median'] = float(h[col].median())
        for col in ['semantic', 'center', 'boundary']:
            if col in h and len(h)>0:
                width = max(1, min(20,len(h)//4))
                history[col+'_mean_first'] = float(h[col].iloc[:width].mean())
                history[col+'_mean_last'] = float(h[col].iloc[-width:].mean())
    report = {'scope':'Existing four validation XY tiles, one specimen; NOT independent validation',
              'ORACLE_WARNING':'Reference instance IDs are used for seeds; oracle results are diagnostic, not achievable test predictions or valid generalization estimates',
              'reference':'Author watershed pseudo-labels; cropped object scores biased; physical spacing unverified',
              'checkpoints': str(ckpt), 'training_sampling': history,
              'ranking':ranking}
    (args.out / 'oracle_decoder_diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2), flush=True)
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--split-run-dir',type=Path,default=Path('runs/segmentation_extended'))
    p.add_argument('--head-run-dir',type=Path,default=Path('runs/instance_heads_v2'))
    p.add_argument('--out',type=Path,default=Path('runs/instance_decoder_oracle'))
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    evaluate(p.parse_args())
