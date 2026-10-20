#!/usr/bin/env python3
"""Exploratory graph reconstruction from trained XYZ voxel affinity predictions.

Uses existing VAL-only crops and author watershed pseudo-labels. This is a
pipeline diagnostic, not independent biological ground truth or a publishable
measure of generalization. No test tile is accessed.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from train_segmentation import read_data
from train_voxel_affinity import make_network, predict_prob, edge_stats, affinity_targets
from instance_watershed_baseline import segment


def graph_from_affinity(fg: np.ndarray, p_same: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """Connected components of positive-affinity edges on foreground voxels.

    This simple graph baseline can drastically merge/split cells. Edges are
    directed in storage (anchored at lower voxel), but the final graph is
    symmetrized and treated as undirected.
    """
    if fg.ndim != 3 or p_same.shape != (3, *fg.shape):
        raise ValueError('Expected fg[Z,Y,X] and affinities[3,Z,Y,X]')
    if not 0 < threshold < 1:
        raise ValueError('threshold must lie strictly between 0 and 1')
    fg = np.asarray(fg, dtype=bool)
    ids = -np.ones(fg.shape, dtype=np.int32)
    n = int(fg.sum())
    if n == 0:
        return np.zeros(fg.shape, np.int32)
    ids[fg] = np.arange(n, dtype=np.int32)
    srcs = []
    dsts = []
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        a = ids[tuple(lo)]
        b = ids[tuple(hi)]
        good = (a >= 0) & (b >= 0) & (p_same[axis][tuple(lo)] >= threshold)
        srcs.append(a[good].astype(np.int32, copy=False))
        dsts.append(b[good].astype(np.int32, copy=False))
    sources = np.concatenate(srcs)
    targets = np.concatenate(dsts)
    if sources.size:
        rr = np.concatenate((sources, targets))
        cc = np.concatenate((targets, sources))
        adjacency = sparse.coo_matrix((np.ones(rr.size, dtype=np.uint8), (rr, cc)), shape=(n, n)).tocsr()
    else:
        adjacency = sparse.csr_matrix((n, n), dtype=np.uint8)
    num, components = connected_components(adjacency, directed=False)
    out = np.zeros(fg.shape, dtype=np.int32)
    out[fg] = components + 1
    return out


def component_stats(pred: np.ndarray) -> dict:
    labels, counts = np.unique(pred[pred > 0], return_counts=True)
    return dict(n_objects=int(len(labels)),
                singleton_fraction=float(np.mean(counts == 1)) if len(counts) else 0.,
                largest_component_voxels=int(counts.max()) if len(counts) else 0,
                largest_component_fraction_of_foreground=float(counts.max() / counts.sum()) if len(counts) else 0.)


def sparse_pq(reference: np.ndarray, predicted: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """IoU>=0.5 one-to-one greedy matching without huge dense NxM matrices."""
    ref_ids, ref_counts = np.unique(reference[reference > 0], return_counts=True)
    pred_ids, pred_counts = np.unique(predicted[predicted > 0], return_counts=True)
    nr, npred = len(ref_ids), len(pred_ids)
    if nr == 0 and npred == 0:
        return dict(n_reference=0, n_predicted=0, tp=0, fp=0, fn=0, pq=1., precision=1., recall=1.)
    both = (reference > 0) & (predicted > 0)
    intersections = []
    if np.any(both):
        rr = np.searchsorted(ref_ids, reference[both])
        pp = np.searchsorted(pred_ids, predicted[both])
        # Can hold pair indices in 64-bit integer even for a large crop.
        joint, count = np.unique(rr.astype(np.int64)*max(1,npred)+pp.astype(np.int64), return_counts=True)
        ridx, pidx = joint // max(1,npred), joint % max(1,npred)
        iou = count / (ref_counts[ridx] + pred_counts[pidx] - count)
        chosen = np.flatnonzero(iou >= iou_threshold)
        chosen = chosen[np.argsort(-iou[chosen])]
        used_r, used_p = set(), set()
        for j in chosen:
            r, p = int(ridx[j]), int(pidx[j])
            if r not in used_r and p not in used_p:
                used_r.add(r); used_p.add(p)
                intersections.append(float(iou[j]))
    tp = len(intersections)
    fp = npred-tp
    fn = nr-tp
    denom = tp+0.5*fp+0.5*fn
    return dict(n_reference=int(nr), n_predicted=int(npred), tp=tp, fp=fp, fn=fn,
                pq=float(sum(intersections)/denom) if denom else 0.,
                precision=tp/max(1,npred), recall=tp/max(1,nr))


def run(args):
    out=args.out
    out.mkdir(parents=True, exist_ok=True)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    audit=json.loads((args.split_run_dir/'data_audit.json').read_text())
    if audit.get('stack_order_check',{}).get('status') != 'FILENAME_ORDER_MATCH':
        raise RuntimeError('Stack-order audit incomplete, do not proceed')
    img, ref, _, _ = read_data(args.data_dir)
    tiles=pd.read_csv(args.split_run_dir/'tile_splits.csv')
    tiles=tiles.loc[tiles['split']=='val'].head(args.max_tiles)
    if tiles.empty:raise RuntimeError('No validation tiles found')
    path=args.affinity_run_dir/'xyz_downweighted.pt'
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if float(ck.get('z_weight',0)) <= 0:
        raise RuntimeError('XYZ-supervised checkpoint required for 3D graph pilot')
    model=make_network(int(ck['base']),args.split_run_dir/'unet_3d.pt')
    model.load_state_dict(ck['state_dict'],strict=True)
    model=model.to(device).eval()
    rows=[]
    for n,t in enumerate(tiles.itertuples(index=False),1):
        y,x,w=int(t.y),int(t.x),int(t.w)
        if w<=2*args.guard:raise ValueError('Guard pixels too large')
        image=img[:,y:y+w,x:x+w]
        label=ref[:,y:y+w,x:x+w]
        with torch.inference_mode():prob=predict_prob(model,image,device)
        g=args.guard
        pp=prob[:, :, g:w-g, g:w-g]
        lab=label[:,g:w-g,g:w-g]
        fg=pp[0]>=0.5
        edg=edge_stats(pp[1:],lab)
        for e in edg:
            rows.append({'tile_y':y,'tile_x':x,'method':'edge_quality',**e})
        ws=segment(pp[0],mode='watershed',threshold=0.5,min_distance=3)
        ws_stats={**sparse_pq(lab,ws), **component_stats(ws)}
        rows.append({'tile_y':y,'tile_x':x,'method':'distance_watershed',**ws_stats})
        graph=graph_from_affinity(fg,pp[1:],args.affinity_threshold)
        graph_stats={**sparse_pq(lab,graph), **component_stats(graph)}
        rows.append({'tile_y':y,'tile_x':x,'method':'affinity_graph',**graph_stats})
        print(f'VAL {n}/{len(tiles)} ({y},{x}): watershed PQ={ws_stats["pq"]:.4f} '
              f'({ws_stats["n_predicted"]} objects), graph PQ={graph_stats["pq"]:.4f} '
              f'({graph_stats["n_predicted"]} objects)', flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'validation_graph_pilot.csv',index=False)
    summary={}
    for method in ['distance_watershed','affinity_graph']:
        block=df[df.method==method]
        summary[method]={'median_pq':float(block.pq.median()),
                         'median_recall':float(block.recall.median()),
                         'median_precision':float(block.precision.median()),
                         'median_n_predicted':float(block.n_predicted.median()),
                         'median_singleton_fraction':float(block.singleton_fraction.median()),
                         'median_largest_component_fraction':float(block.largest_component_fraction_of_foreground.median())}
    eds=df[df.method=='edge_quality'][['tile_y','tile_x','axis','balanced_accuracy','different_recall','different_precision','n_valid','n_different']]
    eds.to_csv(out/'validation_edge_precision.csv',index=False)
    result={'scope':'Already reused VAL tiles of ONE specimen; strictly exploratory. TEST untouched.',
            'pseudo_label_warning':'Author watershed instances, not manual ground truth.',
            'decoder':'Simple 6-neighbor graph components with fixed P(same) threshold; NOT mutex watershed.',
            'model':str(path), 'foreground_threshold':0.5,'affinity_threshold':args.affinity_threshold,
            'max_val_tiles':len(tiles),'summary':summary,
            'limitations':['High local edge balanced accuracy does not guarantee good graph PQ',
                           'Single mistaken high-probability bridge can merge many cells',
                           'Large numbers of low-confidence edges can fragment cells',
                           'Physical spacing and registration still unverified',
                           'Crop-boundary object evaluation biased; validation is repeatedly reused']}
    (out/'graph_pilot.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--split-run-dir',type=Path,required=True)
    p.add_argument('--affinity-run-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--affinity-threshold',type=float,default=0.7)
    p.add_argument('--guard',type=int,default=12)
    p.add_argument('--max-tiles',type=int,default=1)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    run(p.parse_args())

if __name__=='__main__':main()
