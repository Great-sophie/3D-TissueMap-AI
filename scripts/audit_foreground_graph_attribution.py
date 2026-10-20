#!/usr/bin/env python3
"""Counterfactual foreground intervention for voxel affinity region graph.

TRAIN tiles only; uses reference masks in oracle interventions. Not a deployable
model, validation result, method ranking, or independent generalization test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_segmentation import read_data
from train_voxel_affinity import make_network, predict_prob
from evaluate_affinity_region_graph import block_supervoxels, interface_edges, merge_regions
from dynamic_region_merging import build_region_evidence, merge_dynamic_regions
from audit_region_oracle_connectivity import (
    majority_reference_per_supervoxel, oracle_connected_output, graph_masks,
)
from evaluate_affinity_graph_pilot import sparse_pq


def foreground_diagnostics(fg: np.ndarray, ref: np.ndarray) -> dict:
    truth = ref > 0
    tp = int((fg & truth).sum())
    fp = int((fg & ~truth).sum())
    fn = int((~fg & truth).sum())
    return {
        'foreground_tp': tp, 'foreground_fp': fp, 'foreground_fn': fn,
        'foreground_precision': float(tp / max(tp + fp, 1)),
        'foreground_recall': float(tp / max(tp + fn, 1)),
        'foreground_dice': float(2 * tp / max(2 * tp + fp + fn, 1)),
    }


def evaluate_condition(fg: np.ndarray, prob_aff: np.ndarray, reference: np.ndarray,
                       *, block_zyx=(1, 4, 4), z_weight=0.25,
                       score_threshold=0.2, max_negative_fraction=0.35,
                       min_faces=2, max_component_voxels=1200) -> dict:
    """Same frozen affinity probabilities and merger; only foreground changes."""
    fg = np.asarray(fg, dtype=bool)
    reference = np.asarray(reference)
    if fg.ndim != 3 or fg.shape != reference.shape or prob_aff.shape != (3, *fg.shape):
        raise ValueError('Expected aligned fg/reference [Z,Y,X] and affinity [3,Z,Y,X]')
    if any(x < 1 for x in block_zyx):
        raise ValueError('block_zyx must be positive')
    sv = block_supervoxels(fg, block_zyx)
    edges = interface_edges(sv, prob_aff, z_weight=z_weight)
    evidence = build_region_evidence(sv, prob_aff, z_weight=z_weight)
    majority, purity_all, overlap_purity, sizes = majority_reference_per_supervoxel(sv, reference)
    if len(edges) != len(evidence):
        raise AssertionError('Static and dynamic interface enumeration mismatch')

    n = int(sv.max())
    orphan_mask = (majority[1:] == 0)
    orphan_voxels = int(sizes[1:][orphan_mask].sum())
    known_edges = (
        (majority[edges[:,0].astype(np.int64)] > 0)
        & (majority[edges[:,1].astype(np.int64)] > 0)
    ) if len(edges) else np.zeros(0, dtype=bool)
    same_edges = known_edges & (
        majority[edges[:,0].astype(np.int64)] == majority[edges[:,1].astype(np.int64)]
    ) if len(edges) else np.zeros(0, dtype=bool)
    gate = graph_masks(edges, score_threshold, max_negative_fraction, min_faces)

    methods = {}
    static, nstatic = merge_regions(sv, edges, score_threshold, max_negative_fraction,
                                    max_component_voxels, min_faces)
    methods['static_greedy'] = {'metrics': sparse_pq(reference, static), 'n_merges': int(nstatic)}
    dynamic, dinfo = merge_dynamic_regions(
        sv, evidence, score_threshold=score_threshold,
        max_negative_fraction=max_negative_fraction,
        max_component_voxels=max_component_voxels,
        min_faces=min_faces, use_repulsion=True, phase_order='joint')
    methods['dynamic_joint_repulsion'] = {
        'metrics': sparse_pq(reference, dynamic), 'n_merges': int(dinfo['n_merges']),
    }

    unconstrained = majority[sv].astype(np.int32)
    methods['oracle_majority_unconstrained'] = {
        'metrics': sparse_pq(reference, unconstrained),
        'note': 'Reuses same reference ID in disconnected regions, and removes all oracle-unknown regions'
    }
    for name in ('all', 'all_three'):
        connected, nm = oracle_connected_output(sv, edges, majority, gate[name])
        methods['oracle_connected_' + name] = {
            'metrics': sparse_pq(reference, connected), 'n_merges': int(nm),
            'note': 'Reference-guided merges; unknown regions remain predicted objects'
        }

    fg_diag = foreground_diagnostics(fg, reference)
    return {
        **fg_diag,
        'n_supervoxels': n,
        'n_interfaces': int(len(edges)),
        'orphan_supervoxels_without_reference_overlap': int(orphan_mask.sum()),
        'orphan_supervoxel_voxels': orphan_voxels,
        'orphan_fraction_of_predicted_foreground_voxels': float(orphan_voxels / max(int(fg.sum()),1)),
        'oracle_majority_same_interfaces': int(same_edges.sum()),
        'oracle_majority_same_interfaces_kept_by_three_filters': int((same_edges & gate['all_three']).sum()),
        'oracle_majority_same_interface_recall': float((same_edges & gate['all_three']).sum() / max(int(same_edges.sum()),1)),
        'methods': methods,
    }


def inspect_tile(prob: np.ndarray, reference: np.ndarray, **kwargs) -> dict:
    pred = prob[0] >= 0.5
    truth = reference > 0
    result = {
        'predicted_foreground': evaluate_condition(pred, prob[1:], reference, **kwargs),
        'oracle_remove_false_positive_foreground': evaluate_condition(pred & truth, prob[1:], reference, **kwargs),
        'oracle_exact_reference_foreground': evaluate_condition(truth, prob[1:], reference, **kwargs),
    }
    # Fundamental invariant: only the foreground changes; input affinity remains frozen.
    if result['oracle_remove_false_positive_foreground']['foreground_fn'] != result['predicted_foreground']['foreground_fn']:
        raise AssertionError('Removing false positives should not repair foreground false negatives')
    if result['oracle_remove_false_positive_foreground']['foreground_fp'] != 0:
        raise AssertionError('Oracle overlap should have zero false positives')
    if result['oracle_exact_reference_foreground']['foreground_fn'] or result['oracle_exact_reference_foreground']['foreground_fp']:
        raise AssertionError('Reference foreground must be exact')
    return result


def run(args):
    if not args.accept_provisional_registration:
        raise RuntimeError('Acknowledge provisional image registration with --accept-provisional-registration')
    if args.max_tiles < 1 or 2 * args.guard >= 192:
        raise ValueError('Invalid max-tiles or guard')
    audit = json.loads((args.split_run_dir/'data_audit.json').read_text())
    if audit.get('stack_order_check', {}).get('status') != 'FILENAME_ORDER_MATCH':
        raise RuntimeError('Stack order not verified')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')

    img, reference, _, _ = read_data(args.data_dir)
    tiles = pd.read_csv(args.split_run_dir/'tile_splits.csv')
    tiles = tiles[tiles.split == 'train'].head(args.max_tiles)
    if tiles.empty:
        raise RuntimeError('No TRAIN spatial tiles found')
    checkpoint = torch.load(args.affinity_run_dir/'xyz_downweighted.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('z_weight', 0) <= 0:
        raise RuntimeError('Requires XYZ-trained affinity checkpoint')
    model = make_network(int(checkpoint['base']), args.split_run_dir/'unet_3d.pt')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(device).eval()
    rows = []
    reports = []
    args.out.mkdir(parents=True, exist_ok=True)
    for tile in tiles.itertuples(index=False):
        y, x, w = int(tile.y), int(tile.x), int(tile.w)
        g = args.guard
        if 2 * g >= w:
            raise ValueError('Guard too large for this tile')
        with torch.inference_mode():
            p = predict_prob(model, img[:,y:y+w,x:x+w],device)
        p = p[:,:,g:w-g,g:w-g]
        ref = reference[:,y+g:y+w-g,x+g:x+w-g]
        result = inspect_tile(p, ref, block_zyx=(args.region_z,args.region_xy,args.region_xy),
            z_weight=args.z_weight, score_threshold=args.score_threshold,
            max_negative_fraction=args.max_negative_fraction,
            min_faces=args.min_faces, max_component_voxels=args.max_component_voxels)
        reports.append({'tile_y':y,'tile_x':x,'conditions':result})
        for condition, r in result.items():
            for method, mr in r['methods'].items():
                rows.append({'tile_y':y,'tile_x':x,'condition':condition,'method':method,
                    'pq':mr['metrics']['pq'],'tp':mr['metrics']['tp'],
                    'fp':mr['metrics']['fp'],'fn':mr['metrics']['fn'],
                    'n_predicted':mr['metrics']['n_predicted'],
                    'fg_dice':r['foreground_dice'], 'fg_precision':r['foreground_precision'],
                    'fg_recall':r['foreground_recall'],
                    'n_supervoxels':r['n_supervoxels'],'orphan_sv':r['orphan_supervoxels_without_reference_overlap'],
                    'n_merges':mr.get('n_merges')})
        print(f'TRAIN ({y},{x}):',flush=True)
        for condition,r in result.items():
            print(f'  {condition}: FG Dice {r["foreground_dice"]:.4f}, SV {r["n_supervoxels"]}, '
                  f'orphan SV {r["orphan_supervoxels_without_reference_overlap"]}; '
                  f'dynamic PQ {r["methods"]["dynamic_joint_repulsion"]["metrics"]["pq"]:.4f}, '
                  f'oracle-all PQ {r["methods"]["oracle_connected_all"]["metrics"]["pq"]:.4f}', flush=True)
    output = {
        'purpose': 'Reference-foreground counterfactual attribution only; NOT deployable method ranking',
        'scope': 'TRAIN spatial tiles from one specimen; no VAL/TEST accessed',
        'reference': 'Author-produced watershed pseudo-instances, not manual truth',
        'interventions': {
            'predicted_foreground': 'Normal model-predicted foreground',
            'oracle_remove_false_positive_foreground': 'Predicted foreground intersected with reference foreground; removes FP, retains FN',
            'oracle_exact_reference_foreground': 'Uses entire reference foreground; corrects both FP and FN',
        },
        'notes': [
            'The same frozen affinity probabilities, supervoxel shape, thresholds, and merge rules are used across conditions.',
            'Oracle interventions use reference masks during graph construction and must not be deployed or reported as model performance.',
            'Changing foreground can restructure supervoxels/edges, so score differences are counterfactual, not additive causal effects.',
            'Unknown regions are kept by oracle-connected methods but removed by unconstrained majority-label assembly.',
            'Physical spacing and stack registration not verified; TRAIN single specimen, no independent generalization.'
        ],
        'params': {'block_zyx':[args.region_z,args.region_xy,args.region_xy],
                   'z_weight':args.z_weight,'score_threshold':args.score_threshold,
                   'negative_threshold':args.max_negative_fraction, 'min_faces':args.min_faces,
                   'max_component_voxels':args.max_component_voxels},
        'tiles': reports,
    }
    (args.out/'foreground_graph_attribution.json').write_text(json.dumps(output, indent=2, ensure_ascii=False))
    pd.DataFrame(rows).to_csv(args.out/'foreground_graph_comparison.csv',index=False)
    print('Saved:',args.out,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--split-run-dir',type=Path,required=True)
    parser.add_argument('--affinity-run-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-tiles',type=int,default=1)
    parser.add_argument('--region-z',type=int,default=1)
    parser.add_argument('--region-xy',type=int,default=4)
    parser.add_argument('--z-weight',type=float,default=.25)
    parser.add_argument('--score-threshold',type=float,default=.2)
    parser.add_argument('--max-negative-fraction',type=float,default=.35)
    parser.add_argument('--max-component-voxels',type=int,default=1200)
    parser.add_argument('--min-faces',type=int,default=2)
    parser.add_argument('--guard',type=int,default=12)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--accept-provisional-registration',action='store_true')
    run(parser.parse_args())

if __name__ == '__main__':
    main()
