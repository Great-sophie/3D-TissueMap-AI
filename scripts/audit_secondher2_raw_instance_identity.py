from pathlib import Path
import json
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

RUN = Path(
    "runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1"
)
OUT = Path(
    "runs/qc_secondher2_boundary_methods_all92_v0_1"
)
OUT.mkdir(parents=True, exist_ok=True)

summary = json.loads((RUN / "summary.json").read_text())

base = Path(
    "data/inspect/SecondHer2BreastCancerModel/"
    "SecondHer2BreastCancerModel"
)
mask_path = base / (
    "measured_mask_final_segmentation_hwatershed_bg500_90%.tif"
)

x0, x1, y0, y1 = summary["roi_xy"]
zs = summary["z_indices"]

with tifffile.TiffFile(mask_path) as tf:
    raw = np.stack([
        tf.pages[z].asarray()[y0:y1, x0:x1]
        for z in zs
    ])

assert raw.ndim == 3
assert np.issubdtype(raw.dtype, np.integer)
assert np.min(raw) >= 0
assert int(np.max(raw)) < 2**31

raw = raw.astype(np.int32, copy=False)

metrics = pd.read_csv(RUN / "candidate_boundary_metrics.csv")
d = metrics[metrics.method == "nuclear_voronoi"].copy()

source_ids = set(d.source_pseudo_id.astype(int))
raw_ids = set(np.unique(raw[raw > 0]).astype(int))

print("\n=== Raw mask ===")
print("Shape:", raw.shape)
print("Dtype:", raw.dtype)
print("Unique positive IDs:", len(raw_ids))
print("IDs represented in Workflow 12E:", len(source_ids))
print("Workflow IDs missing from raw mask:",
      len(source_ids - raw_ids))
print("Raw IDs without retained seeds:",
      len(raw_ids - source_ids))

assert not (source_ids - raw_ids)

print("\n=== Seed provenance ===")

seed_index = pd.read_csv(
    RUN / "section_nuclear_seed_index.csv"
)

required = {
    "z", "seed_y_local", "seed_x_local",
    "source_pseudo_id"
}

if required.issubset(seed_index.columns):
    z_to_i = {z: i for i, z in enumerate(zs)}
    zi = seed_index.z.map(z_to_i)
    assert zi.notna().all()

    iy = seed_index.seed_y_local.to_numpy(dtype=int)
    ix = seed_index.seed_x_local.to_numpy(dtype=int)

    assert np.all((iy >= 0) & (iy < raw.shape[1]))
    assert np.all((ix >= 0) & (ix < raw.shape[2]))

    observed = raw[zi.to_numpy(dtype=int), iy, ix]
    expected = seed_index.source_pseudo_id.to_numpy(dtype=int)

    print("Seed coordinates checked:", len(expected))
    print("Source-label mismatches:",
          int(np.count_nonzero(observed != expected)))
else:
    print("Coordinate columns unavailable:",
          sorted(required - set(seed_index.columns)))

print("\n=== Raw 3D connectivity ===")

structure6 = ndi.generate_binary_structure(3, 1)
structure26 = ndi.generate_binary_structure(3, 3)

bounds = ndi.find_objects(raw)
records = []

for sid in sorted(source_ids):
    sl = bounds[sid - 1]

    if sl is None:
        raise RuntimeError(f"Source {sid} not found")

    m = raw[sl] == sid

    _, n6 = ndi.label(m, structure6)
    _, n26 = ndi.label(m, structure26)

    occupied = np.flatnonzero(np.any(m, axis=(1, 2)))

    records.append({
        "source_pseudo_id": sid,
        "n_voxels": int(m.sum()),
        "n_sections": len(occupied),
        "internal_z_gaps": (
            int(occupied[-1] - occupied[0] + 1 - len(occupied))
        ),
        "components_6": int(n6),
        "components_26": int(n26),
    })

result = pd.DataFrame(records)

print("Source IDs:", len(result))
print("Multi-section IDs:",
      int((result.n_sections > 1).sum()))
print("6-connected IDs:",
      int((result.components_6 == 1).sum()))
print("26-connected IDs:",
      int((result.components_26 == 1).sum()))
print("IDs with >1 6-connected component:",
      int((result.components_6 > 1).sum()))
print("IDs with internal Z gaps:",
      int((result.internal_z_gaps > 0).sum()))

multi = result[result.n_sections > 1]
print("\n=== Multi-section IDs only ===")
print("Count:", len(multi))
print("6-connected:",
      int((multi.components_6 == 1).sum()))
print("26-connected:",
      int((multi.components_26 == 1).sum()))

output = OUT / "secondher2_raw_instance_identity_qc.csv"
result.to_csv(output, index=False)
print("\nSaved:", output)
