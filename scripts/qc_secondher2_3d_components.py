from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

ROOT = Path(
    "runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1"
)
OUT = Path(
    "runs/qc_secondher2_boundary_methods_all92_v0_1"
)
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "candidate_boundary_metrics.csv")
d = df[df.method == "nuclear_voronoi"].copy()

assert not d.duplicated(["z", "seed_id"]).any()
assert d.source_pseudo_id.notna().all()
assert (d.source_pseudo_id > 0).all()

print("=== Constructing source-ID candidate volume ===")

volume = None

for z in range(92):
    labels = tifffile.imread(
        ROOT / "candidate_labels" /
        f"z{z:03d}_nuclear_voronoi.tif"
    )

    if volume is None:
        volume = np.zeros(
            (92, *labels.shape), dtype=np.int32
        )

    assert labels.shape == volume.shape[1:]

    g = d[d.z == z]

    ids = g.seed_id.to_numpy(dtype=np.int64)
    source_ids = g.source_pseudo_id.to_numpy(
        dtype=np.int32
    )

    observed = np.unique(labels[labels > 0])
    assert np.array_equal(
        np.sort(ids), observed
    ), f"Label mapping mismatch at z={z}"

    lut = np.zeros(
        int(labels.max()) + 1, dtype=np.int32
    )
    lut[ids] = source_ids

    volume[z] = lut[labels]

print("Volume shape:", volume.shape)
print("Unique source IDs:",
      len(np.unique(volume[volume > 0])))

print("\n=== 3D Component Analysis ===")

structure6 = ndi.generate_binary_structure(3, 1)
structure26 = ndi.generate_binary_structure(3, 3)

bounds = ndi.find_objects(volume)

records = []

for sid, g in d.groupby("source_pseudo_id"):
    sid = int(sid)
    sl = bounds[sid - 1]

    if sl is None:
        raise RuntimeError(f"Missing source ID {sid}")

    mask = volume[sl] == sid
    nvox = int(mask.sum())

    lab6, n6 = ndi.label(
        mask, structure=structure6
    )
    lab26, n26 = ndi.label(
        mask, structure=structure26
    )

    sizes6 = np.bincount(
        lab6.ravel()
    )[1:]

    largest_fraction = (
        float(sizes6.max() / nvox)
        if nvox and len(sizes6)
        else np.nan
    )

    records.append({
        "source_pseudo_id": sid,
        "n_fragments": len(g),
        "n_sections": g.z.nunique(),
        "n_voxels_in_roi": nvox,
        "components_6": n6,
        "components_26": n26,
        "connected_6": n6 == 1,
        "connected_26": n26 == 1,
        "largest_component_fraction_6":
            largest_fraction,
    })

result = pd.DataFrame(records)

topology = pd.read_csv(
    OUT / "3d_object_topology_qc.csv"
)

result = result.merge(
    topology[
        [
            "source_pseudo_id",
            "simple_contiguous_candidate",
            "internal_missing_z",
            "n_multifrag_sections",
        ]
    ],
    on="source_pseudo_id",
    validate="one_to_one"
)

simple = (
    result.simple_contiguous_candidate
    .astype(str).str.lower().eq("true")
)

print("\n=== Connectivity Summary ===")
print("Total source IDs:", len(result))
print("6-connected objects:",
      int(result.connected_6.sum()))
print("26-connected objects:",
      int(result.connected_26.sum()))

print("\n=== Simple multi-section candidates ===")
print("Total:", int(simple.sum()))
print("6-connected:",
      int(result.loc[simple, "connected_6"].sum()))
print("26-connected:",
      int(result.loc[simple, "connected_26"].sum()))

print("\n=== Complex candidate topology ===")
complex_mask = ~simple

print("Complex or single-section IDs:",
      int(complex_mask.sum()))
print("6-connected:",
      int(result.loc[complex_mask, "connected_6"].sum()))
print("26-connected:",
      int(result.loc[complex_mask, "connected_26"].sum()))

print("\n=== Fragmentation burden ===")
print("Multiple 6-connected components:",
      int((result.components_6 > 1).sum()))
print("Multiple 26-connected components:",
      int((result.components_26 > 1).sum()))

print("\n=== Largest-component fraction ===")
print(
    result.largest_component_fraction_6.quantile(
        [0.1, 0.25, 0.5, 0.75, 0.9]
    ).to_string()
)

path = OUT / "3d_candidate_component_connectivity.csv"
result.to_csv(path, index=False)

print("\nSaved:", path)
