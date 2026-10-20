#!/usr/bin/env python3
"""Oracle-majority supervoxel *representation* audit (not a model test).

For each connected component within each fixed ZYX grid block, assign the
reference instance ID occupying the most voxels in that atomic supervoxel.
Components assigned to the same reference ID are then identified as a single
predicted instance (an oracle operation, unavailable during real inference).
Report the resulting instance PQ and foreground voxel-label agreement.

IMPORTANT: This is neither a realizable prediction nor a mathematically exact
upper bound on achievable PQ. It quantifies one optimistic majority-label
assembly of reference-foreground supervoxels on a previously used VAL crop.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import tifffile


def atomic_supervoxels(fg: np.ndarray, block: tuple[int, int, int]) -> np.ndarray:
    if fg.ndim != 3 or len(block) != 3 or any(k < 1 for k in block):
        raise ValueError('Expected 3D foreground and positive (Z,Y,X) block sizes')
    out = np.zeros(fg.shape, np.int32)
    next_id = 0
    structure = ndi.generate_binary_structure(3, 1)
    for z in range(0, fg.shape[0], block[0]):
        for y in range(0, fg.shape[1], block[1]):
            for x in range(0, fg.shape[2], block[2]):
                idx = (slice(z,z+block[0]), slice(y,y+block[1]), slice(x,x+block[2]))
                inside = fg[idx]
                if not inside.any():
                    continue
                labels, n = ndi.label(inside, structure=structure)
                if n:
                    target = out[idx]
                    target[inside] = labels[inside] + next_id
                    next_id += int(n)
    return out


def majority_oracle(ref: np.ndarray, sv: np.ndarray) -> tuple[np.ndarray, dict]:
    fg = ref > 0
    if not np.array_equal(fg, sv > 0):
        raise ValueError('Supervoxel foreground and reference foreground differ')
    if not fg.any():
        raise ValueError('Reference foreground is empty')
    combinations, overlap_counts = np.unique(
        np.column_stack((sv[fg].astype(np.int64), ref[fg].astype(np.int64))),
        axis=0, return_counts=True)
    group_ids, start, n_ids = np.unique(combinations[:,0], return_index=True, return_counts=True)
    n_voxels = np.add.reduceat(overlap_counts, start)
    majority_ref = np.zeros(len(group_ids), np.int64)
    majority_size = np.zeros(len(group_ids), np.int64)
    for i, (s, length) in enumerate(zip(start, n_ids)):
        j = s + np.argmax(overlap_counts[s:s+length])
        majority_ref[i] = combinations[j,1]
        majority_size[i] = overlap_counts[j]
    max_sv = int(sv.max())
    mapping = np.zeros(max_sv+1, dtype=np.int64)
    mapping[group_ids] = majority_ref
    reconstructed = mapping[sv]
    n_wrong = int((n_voxels-majority_size).sum())
    return reconstructed, {
        'n_supervoxels': int(len(group_ids)),
        'n_mixed_supervoxels': int(np.sum(n_ids > 1)),
        'n_reference_instances': int(np.unique(ref[fg]).size),
        'minority_voxel_fraction': n_wrong / int(fg.sum()),
        'oracle_majority_voxel_agreement': float((reconstructed[fg] == ref[fg]).mean()),
        'n_reference_ids_remaining': int(np.unique(reconstructed[fg]).size),
    }


def pq_score(ref: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> dict:
    ref_ids, nref = np.unique(ref[ref>0], return_counts=True)
    pred_ids, npred = np.unique(pred[pred>0], return_counts=True)
    nr, np_ = len(ref_ids), len(pred_ids)
    if nr == 0 and np_ == 0:
        return {'pq':1.0,'tp':0,'fp':0,'fn':0,'n_reference':0,'n_predicted':0}
    common = (ref>0)&(pred>0)
    matched_iou=[]
    if common.any():
        rr=np.searchsorted(ref_ids,ref[common]); pp=np.searchsorted(pred_ids,pred[common])
        encoded,counts=np.unique(rr.astype(np.int64)*max(np_,1)+pp.astype(np.int64),return_counts=True)
        ir=encoded//max(np_,1); ip=encoded%max(np_,1)
        iou=counts/(nref[ir]+npred[ip]-counts)
        candidates=np.flatnonzero(iou>=threshold)
        candidates=candidates[np.argsort(-iou[candidates])]
        used_r=set(); used_p=set()
        for j in candidates:
            r,p=int(ir[j]),int(ip[j])
            if r not in used_r and p not in used_p:
                used_r.add(r);used_p.add(p);matched_iou.append(float(iou[j]))
    tp=len(matched_iou);fp=np_-tp;fn=nr-tp
    denom=tp+0.5*(fp+fn)
    return {'pq':float(sum(matched_iou)/denom) if denom else 0.0,
            'tp':tp,'fp':fp,'fn':fn,'n_reference':nr,'n_predicted':np_,
            'precision':tp/max(np_,1),'recall':tp/max(nr,1)}


def analyze(ref: np.ndarray, block: tuple[int,int,int]) -> dict:
    sv=atomic_supervoxels(ref>0, block)
    oracle,stats=majority_oracle(ref,sv)
    return {'block_zyx':list(block),**stats,**{f'oracle_{key}':val for key,val in pq_score(ref,oracle).items()}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tile-y',type=int,default=384)
    ap.add_argument('--tile-x',type=int,default=576)
    ap.add_argument('--tile',type=int,default=192)
    ap.add_argument('--guard',type=int,default=12)
    ap.add_argument('--blocks',default='1x3x3,1x4x4,2x3x3,2x6x6')
    args=ap.parse_args()
    source=next(args.data_dir.glob('full_mask*.tif'),None)
    if source is None:
        ap.error(f'full_mask*.tif not found under {args.data_dir}')
    with tifffile.TiffFile(source) as f:
        volume=f.asarray()
    if volume.ndim!=3:
        raise ValueError(f'Expected ZYX mask, got {volume.shape}')
    y,x,w,g=args.tile_y,args.tile_x,args.tile,args.guard
    if min(y,x,g)>=0 and w>2*g and y+w<=volume.shape[1] and x+w<=volume.shape[2]:
        ref=volume[:,y+g:y+w-g,x+g:x+w-g]
    else:
        raise ValueError('Invalid ROI; check tile, guard and image extent')
    identities=pq_score(ref,ref)
    if abs(identities['pq']-1.0)>1e-9:
        raise AssertionError('Identity PQ must be 1')
    results=[]
    for item in args.blocks.split(','):
        block=tuple(int(v) for v in item.strip().lower().split('x'))
        if len(block)!=3 or any(v<1 for v in block):
            raise ValueError(f'Invalid block {item!r}')
        r=analyze(ref,block)
        results.append(r)
        print(f"block={item} | oracle_majority_agreement={r['oracle_majority_voxel_agreement']:.4f}"
              f" | oracle_PQ={r['oracle_pq']:.4f} | oracle_recall={r['oracle_recall']:.4f}"
              f" | oracle_n_pred={r['oracle_n_predicted']} | mixed={r['n_mixed_supervoxels']}" ,flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    payload={
        'scope':'Single repeatedly reused VALIDATION crop; reference foreground and reference IDs used as oracle',
        'reference':'Author watershed pseudo-labels, NOT independent manual truth',
        'identity_pq':identities['pq'],
        'roi_zyx':list(ref.shape),
        'results':results,
        'critical_limitations':[
            'Oracle-majority assembly uses true reference instance IDs and is NOT a deployable algorithm.',
            'Oracle-majority PQ is NOT a proven mathematical maximum achievable PQ for an atomic partition.',
            'Reference foreground gives an optimistic structural check; predicted-foreground quality may be worse.',
            'Very small supervoxels may improve purity while making learned region merging harder.',
            'The four reused validation tiles cannot provide an unbiased final model-performance comparison.',
            'No verified voxel spacing or manual 3D cell instance ground truth.'
        ]
    }
    (args.out/'oracle_supervoxel_merging.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print('Saved:',args.out/'oracle_supervoxel_merging.json')


if __name__=='__main__':
    main()
