"""Workflow11 v0.1: conservative within-section pseudo-instance signal attribution.

All spatial distances here are in 2-D XY pixels. No physical Z spacing, cell
membrane, or true cell ownership is inferred. Non-mask pixels receive at most
one nearest-nucleus *proxy* owner per slice; ties follow scipy EDT convention.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import distance_transform_edt

REGIONS = ('nuclear_mask', 'perinuclear_1_2px', 'perinuclear_3_5px', 'outer_context_6_10px')
MARKERS = ('Ir193', 'panCK', 'HER2', 'CD31', 'LYVE1')

@dataclass
class SliceOwnership:
    codes: np.ndarray   # flattened dense combined (region, instance_id) codes
    selected: np.ndarray  # flattened pixel indices that belong to a region
    distances: np.ndarray  # 2d EDT distance for diagnostic figure
    regions: np.ndarray  # uint8: nuclear=0, annuli=1..3, unassigned=255
    owners: np.ndarray  # selected owner IDs


def assign_slice(mask2d: np.ndarray, nids: int, limits=(2.0, 5.0, 10.0)) -> SliceOwnership:
    """Return exclusive same-Z XY Voronoi regions, never relabel nuclear pixels."""
    mask2d = np.asarray(mask2d)
    if mask2d.ndim != 2 or not np.issubdtype(mask2d.dtype, np.integer):
        raise ValueError('2D integer mask required')
    if np.any(mask2d < 0) or np.max(mask2d) > nids:
        raise ValueError('mask IDs outside indexed range')
    if not (0.0 < limits[0] < limits[1] < limits[2]):
        raise ValueError('Strictly increasing positive XY radii required')
    reg = np.full(mask2d.shape, 255, np.uint8)
    nuclei = mask2d > 0
    if not np.any(nuclei):
        return SliceOwnership(np.empty(0, np.int64), np.empty(0,np.int64),
                              np.full(mask2d.shape, np.inf, np.float32), reg,
                              np.empty(0, np.int64))
    # Input 0 on all nuclear pixels; EDT finds closest nuclear voxel for each background pixel.
    distance, inds = distance_transform_edt(~nuclei, return_indices=True)
    nearest = mask2d[inds[0], inds[1]]
    reg[nuclei] = 0
    for idx, (left, right) in enumerate(zip((0.0, *limits[:-1]), limits), start=1):
        reg[(~nuclei) & (distance > left) & (distance <= right)] = idx
    sel = np.flatnonzero(reg.ravel() < len(REGIONS))
    region_ids = reg.ravel()[sel].astype(np.int64)
    owner = nearest.ravel()[sel].astype(np.int64)
    if not np.all(owner > 0):
        raise AssertionError('Orphan pixels in assigned region')
    code = region_ids * (nids + 1) + owner
    return SliceOwnership(code, sel, distance, reg, owner)


class RegionAccumulator:
    """Per-marker sums/counts/sumsq/nonzero, streamed one image plane at a time."""
    def __init__(self, max_id: int, markers=MARKERS):
        if max_id < 1:
            raise ValueError('No instance IDs')
        self.max_id = int(max_id)
        self.markers = tuple(markers)
        self.size = (self.max_id + 1) * len(REGIONS)
        self.count = np.zeros(self.size, dtype=np.int64)
        self.n_finite = {m: np.zeros(self.size, np.int64) for m in self.markers}
        self.n_nonzero = {m: np.zeros(self.size, np.int64) for m in self.markers}
        self.sum = {m: np.zeros(self.size, np.float64) for m in self.markers}
        self.sum_sq = {m: np.zeros(self.size, np.float64) for m in self.markers}
        self.seen = 0

    def add_plane(self, ownership: SliceOwnership, images: dict[str, np.ndarray]):
        codes = ownership.codes
        self.count += np.bincount(codes, minlength=self.size)
        for marker in self.markers:
            a = np.asarray(images[marker])
            if a.ndim != 2 or a.size != ownership.regions.size or a.shape != ownership.regions.shape:
                raise ValueError(f'Invalid marker plane shape: {marker}: {a.shape}')
            v = np.asarray(a.ravel()[ownership.selected], dtype=np.float64)
            finite = np.isfinite(v)
            ids = codes[finite]
            values = v[finite]
            self.n_finite[marker] += np.bincount(ids, minlength=self.size)
            self.n_nonzero[marker] += np.bincount(ids, weights=(values != 0).astype(np.int64), minlength=self.size).astype(np.int64)
            self.sum[marker] += np.bincount(ids, weights=values, minlength=self.size)
            self.sum_sq[marker] += np.bincount(ids, weights=values * values, minlength=self.size)
        self.seen += 1

    def stats(self, marker: str):
        n = self.n_finite[marker]
        avg = np.divide(self.sum[marker], n, out=np.full(self.size,np.nan), where=n>0)
        var = np.divide(self.sum_sq[marker], n, out=np.full(self.size,np.nan), where=n>0) - avg*avg
        sd = np.sqrt(np.maximum(var,0))
        fraction_zero = np.divide(n-self.n_nonzero[marker], n, out=np.full(self.size,np.nan), where=n>0)
        return avg.reshape(len(REGIONS),-1), sd.reshape(len(REGIONS),-1), fraction_zero.reshape(len(REGIONS),-1)
