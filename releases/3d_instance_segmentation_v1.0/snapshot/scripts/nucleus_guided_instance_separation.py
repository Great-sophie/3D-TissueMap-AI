#!/usr/bin/env python3
"""Validation-only nucleus-guided 3D cell instance separation.

This is an exploratory research prototype. The only supervised reference is
watershed-derived author labels (not manual cell instances). No physical voxel
spacing is assumed. The script never reads or scores held-out test tiles.

Install in project scripts/ next to train_segmentation.py and
instance_watershed_baseline.py and optimize_instance_separation.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import torch

from train_segmentation import read_data, predict_tile
from instance_watershed_baseline import load_model, compare_instances
from optimize_instance_separation import watershed_instances, split_merge_screen


def _component_coverage(markers: np.ndarray, fg: np.ndarray,
                        signal: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Guarantee at least one seed in every 6-connected FG component.

    Neither this fallback nor overlapping objects guarantee cell correctness.
    """
    comp, ncomp = ndi.label(fg, structure=ndi.generate_binary_structure(3, 1))
    present = np.zeros(ncomp + 1, dtype=bool)
    present[comp[markers > 0]] = True
    missing = np.flatnonzero(~present[1:]) + 1
    if len(missing):
        centers = ndi.maximum_position(signal, labels=comp, index=missing.tolist())
        if len(missing) == 1:
            centers = [centers]
        start = int(markers.max())
        for i, center in enumerate(centers, 1):
            markers[tuple(center)] = start + i
    return markers, int(len(missing)), int(ncomp)


def nucleus_guided_instances(prob: np.ndarray, ir193: np.ndarray,
                              threshold: float = 0.5,
                              intensity_threshold: float = 0.30,
                              min_distance: int = 3,
                              intensity_sigma_zyx: tuple[float, float, float] = (0.6, 1., 1.),
                              distance_sigma: float = 0.8) -> tuple[np.ndarray, dict]:
    """Use nuclear-intensity local maxima to seed EDT-based 3D watershed.

    All smoothing and peak-distance settings are in voxels. Nucleus maxima are
    *candidates*, not validated cell centers or nuclei instances.
    """
    prob = np.asarray(prob, dtype=np.float32)
    ir193 = np.asarray(ir193, dtype=np.float32)
    if prob.ndim != 3 or ir193.shape != prob.shape:
        raise ValueError('Expected identically shaped [Z,Y,X] volumes')
    if not (np.isfinite(prob).all() and np.isfinite(ir193).all()):
        raise ValueError('Image and probability arrays must be finite')
    if not (0 < threshold < 1 and 0 < intensity_threshold < 1):
        raise ValueError('Thresholds must lie strictly between 0 and 1')
    if min_distance < 1:
        raise ValueError('min_distance must be >= 1')
    fg = prob >= threshold
    if not fg.any():
        return np.zeros(prob.shape, dtype=np.int32), dict(n_seeds=0,
            n_intensity_peaks=0, n_fallback_seeds=0, n_foreground_components=0)

    smoothed_intensity = ndi.gaussian_filter(ir193, intensity_sigma_zyx)
    peaks = peak_local_max(smoothed_intensity, labels=fg,
                           threshold_abs=float(intensity_threshold),
                           min_distance=int(min_distance), exclude_border=False)
    markers = np.zeros(prob.shape, dtype=np.int32)
    for k, c in enumerate(peaks, 1):
        markers[tuple(c)] = k

    # Fallback makes result defined for dim/background-only components but does
    # not assert those components contain a real cell.
    markers, nadded, ncomp = _component_coverage(markers, fg, smoothed_intensity)
    distance = ndi.distance_transform_edt(fg)  # voxel units, spacing unknown
    elevation = -ndi.gaussian_filter(distance.astype(np.float32), distance_sigma)
    pred = watershed(elevation, markers=markers, mask=fg).astype(np.int32)
    return pred, dict(n_seeds=int(np.count_nonzero(markers)),
        n_intensity_peaks=int(len(peaks)), n_fallback_seeds=nadded,
        n_foreground_components=ncomp)


def variants():
    return [
        dict(name='distance_baseline', method='distance', min_distance=3, sigma=0.8),
        dict(name='distance_dense', method='distance', min_distance=2, sigma=0.8),
        dict(name='nucleus_peaks_md3', method='nucleus', min_distance=3, intensity_threshold=0.30),
        dict(name='nucleus_peaks_md5', method='nucleus', min_distance=5, intensity_threshold=0.30),
        dict(name='nucleus_peaks_strict', method='nucleus', min_distance=3, intensity_threshold=0.50),
    ]


def _summary(df: pd.DataFrame) -> list[dict]:
    result=[]
    for name, g in df.groupby('variant', sort=False):
        result.append(dict(variant=name, n_validation_tiles=int(len(g)),
            median_pq=float(g.pq.median()), median_recall=float(g.recall.median()),
            median_precision=float(g.precision.median()),
            median_predicted=float(g.n_predicted.median()),
            median_reference=float(g.n_reference.median()),
            total_merge_candidates=int(g.merge_candidate_preds.sum()),
            total_split_candidates=int(g.split_candidate_refs.sum())))
    return sorted(result, key=lambda d: d['median_pq'], reverse=True)


def evaluate(data_dir: Path, run_dir: Path, out: Path,
             device: str = 'auto', guard: int = 12, limit_tiles: int = 0):
    out.mkdir(parents=True, exist_ok=True)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image, ref, _, _ = read_data(data_dir)
    split = pd.read_csv(run_dir/'tile_splits.csv')
    val = split.loc[split['split'].eq('val')]
    if not len(val):
        raise ValueError('No validation tiles in saved split manifest')
    if limit_tiles:
        val = val.iloc[:limit_tiles]
    model = load_model(run_dir/'unet_3d.pt', device)
    rows = []
    for tile_i, t in enumerate(val.itertuples(index=False), 1):
        y,x,w = int(t.y),int(t.x),int(t.w)
        if guard < 0 or guard >= w//2:
            raise ValueError('guard must be nonnegative and less than half tile width')
        tile = dict(y=y, x=x, w=w)
        with torch.inference_mode():
            prob = predict_tile(model, 3, image, tile, device)
        # Use same crop as previous methods: guard is excluded from scores.
        if guard:
            prob = prob[:,guard:-guard,guard:-guard]
            ir = image[:,y+guard:y+w-guard,x+guard:x+w-guard]
            reference = ref[:,y+guard:y+w-guard,x+guard:x+w-guard]
        else:
            ir = image[:,y:y+w,x:x+w]
            reference = ref[:,y:y+w,x:x+w]
        display = {}
        for variant in variants():
            if variant['method'] == 'distance':
                pred, stats = watershed_instances(prob, threshold=0.5,
                    min_distance=variant['min_distance'], sigma=variant['sigma'],
                    ensure_each_component=(variant['name']=='distance_dense'))
            else:
                pred, stats = nucleus_guided_instances(prob, ir, min_distance=variant['min_distance'],
                    intensity_threshold=variant['intensity_threshold'])
            metrics = compare_instances(reference, pred, iou_threshold=0.5)
            topo = split_merge_screen(reference, pred)
            rows.append(dict(tile_y=y, tile_x=x, variant=variant['name'],
                             **metrics, **stats, **topo))
            if tile_i == 1:
                display[variant['name']] = pred
        if tile_i == 1:
            z = int(ir.shape[0] // 2)
            fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
            panels = [ir[z], reference[z],
                      display['distance_baseline'][z],
                      display['distance_dense'][z],
                      display['nucleus_peaks_md3'][z]]
            captions = ['Ir193', 'Author pseudo-reference', 'Distance baseline',
                        'Distance dense', 'Nucleus-guided']
            for a, plane, title in zip(ax,panels,captions):
                a.imshow(plane,cmap='gray' if title=='Ir193' else 'nipy_spectral',
                         interpolation='nearest',vmin=0)
                a.set_title(title, fontsize=9)
                a.axis('off')
            fig.suptitle('Validation only / one specimen / pseudo-reference',fontsize=10)
            fig.tight_layout()
            fig.savefig(out/'nucleus_guided_comparison.png', dpi=150)
            plt.close(fig)
        print(f'Validation tile {tile_i}/{len(val)} y={y} x={x}: complete',flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out/'instance_metrics_validation.csv',index=False)
    rank=_summary(df)
    report=dict(scope='Validation XY tiles, one specimen',
        reference='Author-produced watershed labels, NOT independent manual GT',
        methods='Ir193 intensity-peaks vs foreground EDT seed detection',
        limitations=['Voxel-unit distances only; physical spacing not verified',
            'Cropped instance boundaries bias object scores',
            'Training uses watershed pseudo-labels; comparisons are not independent object truth',
            'Ir193 nuclear signal does not directly delineate full cell boundaries',
            'Validation tuning is exploratory; no independent generalization inference'],
        checkpoint=str(run_dir/'unet_3d.pt'),
        n_validation_tiles=int(len(val)), ranking=rank)
    (out/'nucleus_guided_validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False), encoding='utf-8')
    print('\n=== NUCLEUS-GUIDED VALIDATION RANKING ===')
    print(pd.DataFrame(rank).round(4).to_string(index=False))
    print('Saved:',out)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    parser.add_argument('--run-dir',type=Path,default=Path('runs/segmentation_extended'))
    parser.add_argument('--out',type=Path,default=Path('runs/nucleus_guided_instances'))
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    parser.add_argument('--guard',type=int,default=12)
    parser.add_argument('--limit-tiles',type=int,default=0)
    args=parser.parse_args()
    evaluate(args.data_dir,args.run_dir,args.out,args.device,args.guard,args.limit_tiles)

if __name__=='__main__':
    main()
