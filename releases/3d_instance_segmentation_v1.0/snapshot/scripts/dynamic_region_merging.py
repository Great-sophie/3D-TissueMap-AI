"""Diagnostic dynamic 3D region-adjacency-graph agglomeration.

This is NOT mutex watershed, and repulsion is a boundary-consensus veto rather
than a global must-not-link constraint. No data leakage from reference IDs.

Edges store eight additive sufficient statistics for XY and Z face evidence:
  XY: (weighted_logit_sum, weight_sum, weighted_negative_sum, face_count)
  Z:  (weighted_logit_sum, weight_sum, weighted_negative_sum, face_count)
New region interfaces are formed by summing the *original voxel-face evidence*
for the participating interfaces, not by averaging means of means.
"""
from __future__ import annotations
import heapq
from collections import defaultdict
import numpy as np


def build_region_evidence(sv: np.ndarray, p_same: np.ndarray, z_weight: float = 0.25) -> np.ndarray:
    """Return [a,b,XY_log_sum,XY_weight,XY_neg_weight,XY_faces,Z_log_sum,Z_weight,Z_neg_weight,Z_faces].

    A voxel face at lower index (z/y/x) supplies P(same) in that axis.
    Only contact between different nonzero supervoxels contributes.
    """
    sv = np.asarray(sv)
    p_same = np.asarray(p_same)
    if sv.ndim != 3 or p_same.shape != (3, *sv.shape):
        raise ValueError('Expected sv[Z,Y,X] and p_same[3,Z,Y,X]')
    if not np.issubdtype(sv.dtype, np.integer) or np.any(sv < 0):
        raise ValueError('sv must have nonnegative integer labels')
    if z_weight < 0 or not np.isfinite(z_weight):
        raise ValueError('z_weight must be finite and >= 0')
    if not np.all(np.isfinite(p_same)):
        raise ValueError('Affinity probabilities must be finite')
    if np.any(p_same < 0) or np.any(p_same > 1):
        raise ValueError('Affinity probabilities must belong to [0,1]')

    codebase = int(sv.max()) + 1
    keys, values = [], []
    for axis in range(3):
        if axis == 0 and z_weight == 0:
            continue
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        a = sv[tuple(lo)]; b = sv[tuple(hi)]
        valid = (a > 0) & (b > 0) & (a != b)
        if not np.any(valid):
            continue
        a = a[valid].astype(np.int64); b = b[valid].astype(np.int64)
        key = np.minimum(a, b)*codebase + np.maximum(a, b)
        p = np.clip(p_same[axis][tuple(lo)][valid].astype(np.float64), 1e-5, 1-1e-5)
        w = z_weight if axis == 0 else 1.0
        feature = np.zeros((len(p), 8), np.float64)
        offset = 4 if axis == 0 else 0
        feature[:, offset] = w * np.log(p/(1-p))
        feature[:, offset+1] = w
        feature[:, offset+2] = w * (p < 0.5)
        feature[:, offset+3] = 1.0
        keys.append(key)
        values.append(feature)
    if not keys:
        return np.empty((0, 10), np.float64)
    raw_keys = np.concatenate(keys)
    features = np.concatenate(values, axis=0)
    distinct, inverse = np.unique(raw_keys, return_inverse=True)
    sums = np.stack([np.bincount(inverse, weights=features[:, i], minlength=len(distinct))
                     for i in range(8)], axis=1)
    return np.column_stack((distinct//codebase, distinct%codebase, sums))


def _summary(stat: np.ndarray, phase: str) -> tuple[float, float, int]:
    if phase == 'xy':
        sumlog, weight, neg, faces = stat[:4]
    elif phase == 'joint':
        sumlog, weight, neg, faces = stat[:4]+stat[4:]
    else:
        raise ValueError(f'Unknown phase {phase}')
    if weight <= 0:
        return -np.inf, 1.0, int(faces)
    return float(sumlog/weight), float(neg/weight), int(faces)


def merge_dynamic_regions(
    sv: np.ndarray,
    evidence: np.ndarray,
    *,
    score_threshold: float = 0.2,
    max_negative_fraction: float = 0.35,
    max_component_voxels: int = 1200,
    min_faces: int = 2,
    use_repulsion: bool = True,
    phase_order: str = 'xy_then_z',
) -> tuple[np.ndarray, dict]:
    """Greedy agglomeration with lazy heap updates and *dynamic* interface evidence.

    phase_order='joint' uses XY and Z together from the outset.
    phase_order='xy_then_z' runs XY-only until convergence before enabling Z.
    Score gating in every phase means this does not guarantee fewer components
    than static merging; reducing oversegmentation is a hypothesis to test.
    """
    sv = np.asarray(sv)
    evidence = np.asarray(evidence, dtype=np.float64)
    if sv.ndim != 3 or not np.issubdtype(sv.dtype, np.integer) or np.any(sv < 0):
        raise ValueError('sv must be a nonnegative 3D integer array')
    if evidence.ndim != 2 or evidence.shape[1] != 10:
        raise ValueError('Expected edge evidence [N,10]')
    if not np.all(np.isfinite(evidence)):
        raise ValueError('Edge evidence must be finite')
    if min_faces < 1 or max_component_voxels < 1:
        raise ValueError('min_faces and max_component_voxels must be positive')
    if not (0 <= max_negative_fraction <= 1):
        raise ValueError('max_negative_fraction must be in [0,1]')
    if phase_order not in ('joint', 'xy_then_z'):
        raise ValueError('phase_order must be joint or xy_then_z')

    n = int(sv.max())
    sizes = np.bincount(sv.ravel(), minlength=n+1).astype(np.int64)
    active = sizes > 0
    active[0] = False
    parent = np.arange(n+1, dtype=np.int32)
    neighbors: list[set[int]] = [set() for _ in range(n+1)]
    edges: dict[tuple[int, int], np.ndarray] = {}

    def ordered(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    for row in evidence:
        a,b = int(row[0]), int(row[1])
        if not (0 < a <= n and 0 < b <= n) or a == b or not active[a] or not active[b]:
            raise ValueError('Edge node IDs do not match nonempty supervoxels')
        key=ordered(a,b)
        if key in edges:
            raise ValueError('Evidence contains duplicate interfaces')
        edges[key] = row[2:].copy()
        neighbors[a].add(b)
        neighbors[b].add(a)
    initial_edges=len(edges)
    stats={'initial_supervoxels':int(active.sum()), 'initial_interfaces':initial_edges,
           'phase_order':phase_order, 'use_repulsion':bool(use_repulsion),
           'score_threshold':float(score_threshold),
           'max_negative_fraction':float(max_negative_fraction),
           'min_faces':int(min_faces), 'max_component_voxels':int(max_component_voxels),
           'n_merges':0, 'phase_stats':[]}
    phases = ('joint',) if phase_order == 'joint' else ('xy', 'joint')

    for phase in phases:
        heap = []
        tickets: dict[tuple[int,int], int] = {}
        nonce = 0
        info = {'phase':phase, 'merges':0, 'heap_pops':0, 'blocked_by_volume_cap':0,
                'stale_events':0, 'candidate_events':0}

        def enqueue(key: tuple[int,int]):
            nonlocal nonce
            if key not in edges:
                return
            score, negfrac, faces = _summary(edges[key], phase)
            if faces < min_faces or score < score_threshold:
                tickets.pop(key, None)
                return
            if use_repulsion and negfrac > max_negative_fraction:
                tickets.pop(key, None)
                return
            nonce+=1
            tickets[key]=nonce
            heapq.heappush(heap, (-score, nonce, *key))
            info['candidate_events'] += 1

        for key in sorted(edges):
            enqueue(key)
        while heap:
            _, token, a, b = heapq.heappop(heap)
            info['heap_pops']+=1
            key=(a,b)
            if tickets.get(key) != token or not active[a] or not active[b]:
                info['stale_events']+=1
                continue
            if sizes[a]+sizes[b] > max_component_voxels:
                info['blocked_by_volume_cap']+=1
                tickets.pop(key, None)
                # Volume can only grow; no need to reconsider this pair unless its
                # neighbor interface changes, in which case enqueue rechecks cap.
                continue
            # Merge smaller into larger to reduce adjacency updates.
            if sizes[b] > sizes[a]:
                a,b=b,a
            # a and b remain root IDs; map on success, never relabel SV itself.
            parent[b]=a
            sizes[a]+=sizes[b]
            active[b]=False
            info['merges']+=1
            stats['n_merges']+=1
            shared=ordered(a,b)
            edges.pop(shared, None)
            tickets.pop(shared, None)
            neighbors[a].discard(b)
            neighbors[b].discard(a)
            for k in sorted(neighbors[b]):
                oldkey=ordered(b,k)
                value=edges.pop(oldkey)
                tickets.pop(oldkey, None)
                neighbors[k].discard(b)
                if k==a:
                    continue
                target=ordered(a,k)
                if target in edges:
                    value=value+edges.pop(target)
                    tickets.pop(target, None)
                edges[target]=value
                neighbors[a].add(k)
                neighbors[k].add(a)
                enqueue(target)
            neighbors[b].clear()
        stats['phase_stats'].append(info)

    active_labels=np.flatnonzero(active)
    remap=np.zeros(n+1, np.int32)
    remap[active_labels]=np.arange(1,len(active_labels)+1,dtype=np.int32)
    # Resolve potentially multi-level parent pointers for every original label.
    for i in np.unique(sv[sv>0]):
        k=int(i)
        while parent[k]!=k:
            k=int(parent[k])
        remap[i]=remap[k]
    stats['final_components']=int(len(active_labels))
    stats['remaining_interfaces']=int(len(edges))
    return remap[sv], stats
