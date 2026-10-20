#!/usr/bin/env python3
"""Validation-only 3D cell-instance error analysis and seed optimization.

Place next to train_segmentation.py and instance_watershed_baseline.py.
Uses author watershed pseudo-labels from one specimen. Never accesses test tiles.
All distances are in voxels; physical spacing is unverified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import torch

from train_segmentation import read_data, predict_tile
from instance_watershed_baseline import load_model, compare_instances


def split_merge_screen(ref: np.ndarray, pred: np.ndarray,
                       minimum_voxels: int = 10, minimum_ref_share: float = 0.15) -> dict:
    """Count candidate mergers/splits using overlap links, not verified topology events.

    A link exists when a predicted object intersects >=10 voxels and >=15% of
    a reference object's voxels. Several refs linked to one pred -> merger
    candidate. Several preds linked to one ref -> split candidate. Cropped
    edges and imperfect pseudo-labels can cause spurious associations.
    """
    r, p = np.asarray(ref), np.asarray(pred)
    if r.shape != p.shape:
        raise ValueError('Reference/prediction shapes differ')
    mask = (r > 0) & (p > 0)
    if not mask.any():
        return dict(n_links=0, merge_candidate_preds=0, split_candidate_refs=0)
    ri, rc = np.unique(r[r > 0], return_counts=True)
    pairs, overlap = np.unique(np.stack([r[mask], p[mask]], axis=1),
                               axis=0, return_counts=True)
    sizes = rc[np.searchsorted(ri, pairs[:, 0])]
    keep = (overlap >= minimum_voxels) & (overlap / sizes >= minimum_ref_share)
    links = pairs[keep]
    if len(links) == 0:
        return dict(n_links=0, merge_candidate_preds=0, split_candidate_refs=0)
    _, preds_per_ref = np.unique(links[:, 0], return_counts=True)
    _, refs_per_pred = np.unique(links[:, 1], return_counts=True)
    return dict(n_links=int(len(links)),
                merge_candidate_preds=int(np.sum(refs_per_pred >= 2)),
                split_candidate_refs=int(np.sum(preds_per_ref >= 2)))


def watershed_instances(prob: np.ndarray, *, min_distance: int = 3,
                        sigma: float = 0.8, threshold: float = 0.5,
                        ensure_each_component: bool = False) -> tuple[np.ndarray, dict]:
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 3 or not np.all(np.isfinite(prob)):
        raise ValueError('Expected finite 3D probability volume')
    if not 0 < threshold < 1 or min_distance < 1 or sigma < 0:
        raise ValueError('Invalid watershed parameters')
    fg = prob >= threshold
    if not fg.any():
        return np.zeros_like(fg, dtype=np.int32), dict(n_seeds=0,n_added_seeds=0,n_components=0)
    edt = ndi.distance_transform_edt(fg)  # voxel units only
    smoothed = ndi.gaussian_filter(edt.astype(np.float32), sigma=sigma) if sigma else edt
    coords = peak_local_max(smoothed, min_distance=min_distance,
                            threshold_abs=1.0, labels=fg, exclude_border=False)
    markers = np.zeros_like(fg, dtype=np.int32)
    for k, idx in enumerate(coords, 1):
        markers[tuple(idx)] = k
    cc, ncc = ndi.label(fg, structure=ndi.generate_binary_structure(3, 1))
    added = 0
    if ensure_each_component:
        # Find components with no maxima. Use a single vectorized labeled
        # maximum lookup rather than scanning the full volume for each ID.
        component_has_seed = np.zeros(ncc + 1, dtype=bool)
        component_has_seed[cc[markers > 0]] = True
        missing = np.flatnonzero(~component_has_seed[1:]) + 1
        if len(missing):
            centers = ndi.maximum_position(edt, labels=cc, index=missing.tolist())
            if len(missing) == 1:
                centers = [centers]
            for center in centers:
                markers[tuple(center)] = len(coords) + added + 1
                added += 1
    if not np.any(markers):
        markers, _ = ndi.label(fg, structure=ndi.generate_binary_structure(3, 1))
    result = watershed(-smoothed, markers=markers, mask=fg).astype(np.int32)
    return result, dict(n_seeds=int(np.count_nonzero(markers)),
                        n_added_seeds=int(added), n_components=int(ncc))


def variants():
    # Baseline matches the previously used configuration exactly.
    return [
        dict(name='baseline',min_distance=3,sigma=0.8,ensure_each_component=False),
        dict(name='seed_coverage',min_distance=3,sigma=0.8,ensure_each_component=True),
        dict(name='seed_dense',min_distance=2,sigma=0.8,ensure_each_component=True),
        dict(name='seed_sparse',min_distance=4,sigma=0.8,ensure_each_component=True),
        dict(name='smooth_low',min_distance=3,sigma=0.5,ensure_each_component=True),
        dict(name='smooth_high',min_distance=3,sigma=1.2,ensure_each_component=True),
    ]


def summarize(df: pd.DataFrame) -> list[dict]:
    rows = []
    for name, sub in df.groupby('variant', sort=False):
        rows.append(dict(variant=name,n_tiles=int(len(sub)),
                         median_pq=float(sub.pq.median()),
                         median_recall=float(sub.recall.median()),
                         median_precision=float(sub.precision.median()),
                         median_predicted=int(sub.n_predicted.median()),
                         median_reference=int(sub.n_reference.median()),
                         total_merge_candidates=int(sub.merge_candidate_preds.sum()),
                         total_split_candidates=int(sub.split_candidate_refs.sum())))
    return sorted(rows, key=lambda a:a['median_pq'], reverse=True)


def run(data_dir: Path, run_dir: Path, out: Path, device='auto',
        threshold: float = 0.5, guard: int = 12, limit_tiles: int = 0):
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between 0 and 1')
    out.mkdir(parents=True,exist_ok=True)
    cache = out / 'validation_probability_cache'
    cache.mkdir(exist_ok=True)
    # Do not silently reuse old probabilities when the checkpoint, source
    # reference or tile split has changed. A new run directory is safest.
    checkpoint = run_dir/'unet_3d.pt'
    split_path = run_dir/'tile_splits.csv'
    full_masks = list(data_dir.glob('full_mask*.tif'))
    archives = list(data_dir.glob('*SIMILARITY*.zip'))
    if len(full_masks) != 1 or len(archives) != 1:
        raise FileNotFoundError('Expected exactly one full mask and one registered-image ZIP')
    signature = dict(mask_size=full_masks[0].stat().st_size,
                     mask_mtime_ns=full_masks[0].stat().st_mtime_ns,
                     image_archive_size=archives[0].stat().st_size,
                     image_archive_mtime_ns=archives[0].stat().st_mtime_ns,
                     checkpoint=str(checkpoint.resolve()),
                     checkpoint_size=checkpoint.stat().st_size,
                     checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
                     split_size=split_path.stat().st_size,
                     split_mtime_ns=split_path.stat().st_mtime_ns,
                     data_dir=str(data_dir.resolve()),guard=guard)
    signature_path = cache/'signature.json'
    if signature_path.exists():
        old=json.loads(signature_path.read_text(encoding='utf-8'))
        if old != signature:
            raise RuntimeError('Cached validation probabilities were produced by different inputs. Delete the validation_probability_cache folder or choose a new --out.')
    elif any(cache.glob('*.npz')):
        raise RuntimeError('Cached probabilities lack provenance signature; choose a new --out or delete cache.')
    else:
        signature_path.write_text(json.dumps(signature,indent=2),encoding='utf-8')
    img, reference, _, _ = read_data(data_dir)
    manifest = pd.read_csv(split_path)
    tiles = manifest.loc[manifest['split'].eq('val')].copy()
    if len(tiles) == 0:
        raise ValueError('No validation tiles in manifest')
    if limit_tiles > 0:
        tiles = tiles.iloc[:limit_tiles]
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = None
    rows = []
    for i, t in enumerate(tiles.itertuples(index=False), 1):
        y, x, w = int(t.y), int(t.x), int(t.w)
        if w <= 2*guard:
            raise ValueError('guard is too large')
        cached = cache/f'val_y{y}_x{x}_w{w}.npz'
        if cached.exists():
            with np.load(cached) as d:
                prob, ref = d['prob'].astype(np.float32),d['ref'].astype(np.int32)
        else:
            if model is None:
                model=load_model(run_dir/'unet_3d.pt',device)
            with torch.inference_mode():
                prob = predict_tile(model,3,img,dict(y=y,x=x,w=w),device)
            prob = prob[:,guard:w-guard,guard:w-guard]
            ref = reference[:,y+guard:y+w-guard,x+guard:x+w-guard]
            np.savez_compressed(cached,prob=prob.astype(np.float32),ref=ref)
        for variant in variants():
            pred, stat = watershed_instances(prob, threshold=threshold,
                    min_distance=variant['min_distance'],sigma=variant['sigma'],
                    ensure_each_component=variant['ensure_each_component'])
            metric = compare_instances(ref, pred, iou_threshold=0.5)
            topo = split_merge_screen(ref,pred)
            rows.append(dict(tile_y=y,tile_x=x,variant=variant['name'],
                        **metric,**stat,**topo))
        print(f'Validation tile {i}/{len(tiles)} y={y} x={x} complete',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'instance_variant_metrics_validation.csv',index=False)
    summary=summarize(df)
    report=dict(scope='validation only; one specimen',
                dataset='LVILymph real IMC, author watershed pseudo-reference',
                baseline_checkpoint=str(run_dir/'unet_3d.pt'),
                threshold=threshold, physical_spacing='UNVERIFIED; all distances in voxels',
                boundary_bias='instances truncated by XY guard/crop and volume ends',
                split_merge_definition='Candidates: >=2 links with >=10 overlapping voxels and >=15% of each reference volume; not confirmed errors',
                variants=variants(),ranking=summary,
                selection_warning='Do not use held-out test tiles for selection; this small validation within one specimen may be overfit.')
    (out/'instance_optimization_validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n=== VALIDATION RANKING (diagnostic, not patient generalization) ===')
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print('Saved:',out)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/inspect/LVILymph'))
    p.add_argument('--run-dir',type=Path,default=Path('runs/segmentation_extended'))
    p.add_argument('--out',type=Path,default=Path('runs/instance_optimization'))
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threshold',type=float,default=0.5)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--limit-tiles',type=int,default=0,help='Quick smoke test on 1 tile; default 0 runs all validation tiles')
    a=p.parse_args()
    run(a.data_dir,a.run_dir,a.out,a.device,a.threshold,a.guard,a.limit_tiles)

if __name__=='__main__':
    main()
