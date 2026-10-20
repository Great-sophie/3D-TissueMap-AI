from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree

RUN = Path('runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1')
PROTEIN = Path('runs/secondher2_3d_object_protein_quantification_v0_1/3d_her2_object_quantification.csv')
OUT = Path('runs/secondher2_3d_spatial_discordance_v0_1')
if OUT.exists():
    raise FileExistsError(f'Refusing to overwrite: {OUT}')

meta = json.loads((RUN / 'summary.json').read_text())
x0, x1, y0, y1 = meta['roi_xy']
base = Path('data/inspect/SecondHer2BreastCancerModel/SecondHer2BreastCancerModel')
mask = base / 'measured_mask_final_segmentation_hwatershed_bg500_90%.tif'
obj = pd.read_csv(PROTEIN)
assert obj.source_pseudo_id.is_unique

# Per-voxel centroid sums in micrometres (OME: X=1, Y=1, Z=2 um).
records = []
with tifffile.TiffFile(mask) as tf:
    for z in meta['z_indices']:
        lab = tf.pages[z].asarray()[y0:y1, x0:x1]
        yy, xx = np.nonzero(lab)
        if not len(xx):
            continue
        ids, inv = np.unique(lab[yy, xx], return_inverse=True)
        n = np.bincount(inv)
        sx = np.bincount(inv, weights=x0 + xx + 0.5)
        sy = np.bincount(inv, weights=y0 + yy + 0.5)
        sz = n * (z + 0.5) * 2.0
        records.extend(zip(ids.tolist(), n.tolist(), sx.tolist(), sy.tolist(), sz.tolist()))

centroids = pd.DataFrame(records, columns=['source_pseudo_id', 'n', 'sx', 'sy', 'sz'])
centroids = centroids.groupby('source_pseudo_id', as_index=False).sum()
centroids['x_um'] = centroids.sx / centroids.n
centroids['y_um'] = centroids.sy / centroids.n
centroids['z_um'] = centroids.sz / centroids.n
centroids = centroids[['source_pseudo_id', 'n', 'x_um', 'y_um', 'z_um']]
obj = obj.merge(centroids, on='source_pseudo_id', validate='one_to_one')
assert len(obj) == 13483
assert (obj.n == obj.n_voxels).all(), 'Mask-vs-protein voxel count mismatch'
assert obj[['x_um','y_um','z_um']].notna().all().all()

obj['group'] = np.where(
    obj.retained_by_workflow12e.astype(str).str.lower().eq('true'),
    'retained', 'excluded'
)
# Percentile ranks are estimated separately within each group.
obj['within_group_rank3B5'] = obj.groupby('group')['HER2_3B5_mean'].rank(pct=True)
obj['within_group_rank29D8'] = obj.groupby('group')['HER2_29D8_mean'].rank(pct=True)
obj['rank_discordance'] = obj.within_group_rank3B5 - obj.within_group_rank29D8

rng = np.random.default_rng(20261019)
K, B = 12, 199

def moran(values, near):
    v = np.asarray(values, float)
    v = v - v.mean()
    denom = np.sum(v * v)
    if denom <= 0:
        return float('nan')
    return float(np.sum(v * v[near].mean(axis=1)) / denom)

results = []
for name, g in obj.groupby('group'):
    points = g[['x_um','y_um','z_um']].to_numpy(float)
    assert len(points) > K + 2
    dists, near_raw = cKDTree(points).query(points, k=K + 2)
    near = np.vstack([
        idx[idx != i][:K]
        for i, idx in enumerate(np.atleast_2d(near_raw))
    ])
    # Actual distance of the 12th distinct object neighbour.
    radius = np.linalg.norm(points[:,None,:] - points[near], axis=2).max(axis=1)
    v = g['rank_discordance'].to_numpy(float)
    observed = moran(v, near)
    null = np.array([moran(rng.permutation(v), near) for _ in range(B)])
    p_two_sided = (1 + np.count_nonzero(
        np.abs(null - null.mean()) >= abs(observed - null.mean())
    )) / (B + 1)
    results.append({
        'group': name,
        'n_objects': len(g),
        'clone_spearman': g.HER2_3B5_mean.corr(g.HER2_29D8_mean, method='spearman'),
        'median_abs_rank_discordance': np.median(np.abs(v)),
        'median_12nn_radius_um': np.median(radius),
        'moran_I_rank_discordance': observed,
        'shuffle_null_mean': null.mean(),
        'shuffle_p_two_sided_descriptive': p_two_sided,
    })

OUT.mkdir(parents=True)
obj.to_csv(OUT / '3d_her2_object_spatial_coordinates.csv', index=False)
pd.DataFrame(results).to_csv(OUT / '3d_spatial_discordance_summary.csv', index=False)
print('\n=== 3D SPATIAL HER2 DISCORDANCE ===')
print(pd.DataFrame(results).round(4).to_string(index=False))
print('\nSUCCESS:', OUT)
print('Coordinates use OME nominal micrometre pixel spacing, not validated cell centers.')
