#!/usr/bin/env python3
"""Diagnostic-only affinity graph threshold and implementation audit.

Uses ONE previously reused VAL tile of one specimen; no new training or TEST.
Expected to live next to evaluate_affinity_graph_pilot.py in scripts/.
Outputs are NOT valid generalization estimates and must not be used to tune
and report a 'best' threshold on the same repeatedly reused VAL tile.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train_segmentation import read_data
from train_voxel_affinity import make_network, predict_prob, affinity_targets
from evaluate_affinity_graph_pilot import graph_from_affinity, sparse_pq, component_stats

AXES = ('Z', 'Y', 'X')

def inspect_one(fg, reference, p_same, thresholds):
    assert fg.shape == reference.shape and p_same.shape == (3, *fg.shape)
    edges = []
    for axis, name in enumerate(AXES):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        a, b = fg[tuple(lo)], fg[tuple(hi)]
        edge_prob = p_same[axis][tuple(lo)]
        pred_valid = a & b
        vals = edge_prob[pred_valid]
        ra, rb = reference[tuple(lo)], reference[tuple(hi)]
        conditional = pred_valid & (ra > 0) & (rb > 0)
        same = conditional & (ra == rb)
        diff = conditional & (ra != rb)
        def quant(a):
            return [float(v) for v in np.quantile(a, [0.1, 0.5, 0.9])] if a.size else [None]*3
        entry = {
            'axis': name,
            'n_predicted_foreground_edges': int(pred_valid.sum()),
            'n_reference_same_within_pred_fg': int(same.sum()),
            'n_reference_different_within_pred_fg': int(diff.sum()),
            'p_same_quantiles_pred_foreground_edges_p10_p50_p90': quant(vals),
            'p_same_quantiles_ref_same_p10_p50_p90': quant(edge_prob[same]),
            'p_same_quantiles_ref_different_p10_p50_p90': quant(edge_prob[diff]),
        }
        for t in thresholds:
            key = f'{t:.2f}'
            entry[f'fraction_pred_fg_edges_kept_at_{key}'] = float(np.mean(vals >= t)) if vals.size else 0.
        edges.append(entry)
    rows = []
    for t in thresholds:
        reconstructed = graph_from_affinity(fg, p_same, t)
        stats = sparse_pq(reference, reconstructed)
        c = component_stats(reconstructed)
        rows.append({'threshold': t, 'pq_diagnostic_only': stats['pq'],
                     'n_reference': stats['n_reference'], 'n_predicted': stats['n_predicted'],
                     'tp': stats['tp'], 'precision': stats['precision'], 'recall': stats['recall'],
                     'singleton_fraction': c['singleton_fraction'],
                     'largest_component_fraction_of_foreground': c['largest_component_fraction_of_foreground']})
    # Oracle same-instance edges verify correct direction, indexing and graph construction.
    oracle_same, _ = affinity_targets(reference)
    oracle = graph_from_affinity(reference > 0, oracle_same, 0.5)
    oracle_metrics = sparse_pq(reference, oracle)
    return edges, rows, oracle_metrics

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--split-run-dir', required=True, type=Path)
    p.add_argument('--affinity-run-dir', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--thresholds', default='0.40,0.50,0.60,0.70')
    p.add_argument('--tile-index', type=int, default=0)
    p.add_argument('--guard', type=int, default=12)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = p.parse_args()
    thresholds = [float(i.strip()) for i in args.thresholds.split(',')]
    if not thresholds or any(t <= 0 or t >= 1 for t in thresholds) or len(set(thresholds)) != len(thresholds):
        p.error('Thresholds must be unique numbers strictly between 0 and 1')
    if len(thresholds) > 8:
        p.error('Use at most 8 diagnostic thresholds')
    if args.tile_index < 0:
        p.error('Tile index must be nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.split_run_dir/'tile_splits.csv')
    v = manifest[manifest['split']=='val'].reset_index(drop=True)
    if args.tile_index >= len(v):
        p.error(f'tile-index must be < {len(v)}')
    audit = json.loads((args.split_run_dir/'data_audit.json').read_text())
    if audit.get('stack_order_check',{}).get('status') != 'FILENAME_ORDER_MATCH':
        raise RuntimeError('Stack order audit not completed; stop')
    image, reference, _, _ = read_data(args.data_dir)
    row = v.iloc[args.tile_index]
    y,x,w = int(row['y']), int(row['x']), int(row['w'])
    if w <= args.guard*2:
        raise ValueError('Guard too large')
    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if dev == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    ck = torch.load(args.affinity_run_dir/'xyz_downweighted.pt', map_location='cpu', weights_only=True)
    if float(ck['z_weight']) <= 0:
        raise RuntimeError('XYZ trained checkpoint required')
    net = make_network(int(ck['base']), args.split_run_dir/'unet_3d.pt')
    net.load_state_dict(ck['state_dict'], strict=True)
    net = net.to(dev).eval()
    with torch.inference_mode():
        probs = predict_prob(net, image[:,y:y+w,x:x+w], dev)
    g = args.guard
    pr = probs[:,:,g:w-g,g:w-g]
    ref = reference[:,y+g:y+w-g,x+g:x+w-g]
    fg = pr[0] >= 0.5
    edges, runs, oracle = inspect_one(fg, ref, pr[1:], thresholds)
    edf, rdf = pd.DataFrame(edges), pd.DataFrame(runs)
    edf.to_csv(args.out/'edge_probability_diagnostics.csv', index=False)
    rdf.to_csv(args.out/'threshold_response_diagnostic.csv', index=False)
    result = {
        'scope': 'One previously reused VAL tile; no TEST. Troubleshooting only, NOT threshold selection or generalization',
        'tile_xy': [y,x],
        'foreground_pred_fraction': float(fg.mean()),
        'foreground_reference_fraction': float((ref>0).mean()),
        'thresholds': thresholds,
        'oracle_graph_reference_fg_reference_affinity': oracle,
        'oracle_interpretation': 'If oracle graph PQ<1, identical labels may form disconnected islands within this crop; investigate, not automatically a graph implementation error.',
        'edge_probability_by_axis': edges,
        'graph_diagnostics': runs,
        'limits': ['Only author watershed pseudo labels', 'Reused VAL tile', 'Pixel/voxel geometry; spacing unverified',
                   'Threshold scan is diagnosis only: do not choose/report test performance based on this VAL tile'],
    }
    (args.out/'graph_fragmentation_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('=== ORACLE SAME-ID AFFINITY (REFERENCE FG) ===')
    print('PQ:', round(oracle['pq'],4), 'Ref:', oracle['n_reference'], 'Pred:', oracle['n_predicted'])
    print('\n=== EDGES AT EACH THRESHOLD ===')
    keep = ['axis','n_predicted_foreground_edges', 'p_same_quantiles_pred_foreground_edges_p10_p50_p90']
    keep += [f'fraction_pred_fg_edges_kept_at_{t:.2f}' for t in thresholds]
    print(edf[keep].to_string(index=False))
    print('\n=== COMPONENT DIAGNOSTICS (NOT MODEL RANKING) ===')
    print(rdf.round(4).to_string(index=False))
    print('\nSaved:', args.out)

if __name__ == '__main__':
    main()
