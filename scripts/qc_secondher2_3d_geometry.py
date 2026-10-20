from pathlib import Path
import numpy as np
import pandas as pd
import tifffile

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

# Calculate 2D region centroids, in ROI pixel coordinates.
frames = []

for z in sorted(d.z.unique()):
    labels = tifffile.imread(
        ROOT / "candidate_labels" /
        f"z{z:03d}_nuclear_voronoi.tif"
    )

    h, w = labels.shape
    y, x = np.indices((h, w))

    flat = labels.ravel().astype(np.int64)
    count = np.bincount(flat)
    sx = np.bincount(flat, weights=x.ravel(),
                     minlength=len(count))
    sy = np.bincount(flat, weights=y.ravel(),
                     minlength=len(count))

    g = d[d.z == z].copy()
    ids = g.seed_id.to_numpy(dtype=np.int64)

    assert ids.min() > 0
    assert ids.max() < len(count)
    assert (count[ids] > 0).all()
    assert np.array_equal(
        np.sort(np.unique(flat[flat > 0])),
        np.sort(ids)
    )
    assert np.array_equal(
        count[ids],
        g.segment_area_px.to_numpy()
    )

    g["centroid_x_px"] = sx[ids] / count[ids]
    g["centroid_y_px"] = sy[ids] / count[ids]
    frames.append(g)

geo = pd.concat(frames, ignore_index=True)

track_rows = []
transition_rows = []

for sid, g in geo.groupby("source_pseudo_id"):
    g = g.sort_values(["z", "seed_id"])

    sizes = g.groupby("z").size()
    zs = sizes.index.to_numpy(dtype=int)

    n_sections = len(zs)
    n_multifrag_sections = int((sizes > 1).sum())
    missing_z = int(zs[-1] - zs[0] + 1 - n_sections)

    simple_topology = (
        n_sections >= 2
        and missing_z == 0
        and n_multifrag_sections == 0
    )

    track_rows.append({
        "source_pseudo_id": sid,
        "n_fragments": len(g),
        "n_sections": n_sections,
        "z_min": int(zs[0]),
        "z_max": int(zs[-1]),
        "internal_missing_z": missing_z,
        "n_multifrag_sections": n_multifrag_sections,
        "simple_contiguous_candidate": simple_topology,
    })

    # Only evaluate unambiguous 1-to-1 adjacent transitions.
    for z1, z2 in zip(zs[:-1], zs[1:]):
        if z2 != z1 + 1:
            continue

        a = g[g.z == z1]
        b = g[g.z == z2]

        if len(a) != 1 or len(b) != 1:
            continue

        a = a.iloc[0]
        b = b.iloc[0]

        dist = np.hypot(
            b.centroid_x_px - a.centroid_x_px,
            b.centroid_y_px - a.centroid_y_px
        )

        transition_rows.append({
            "source_pseudo_id": sid,
            "z_from": int(z1),
            "z_to": int(z2),
            "seed_from": int(a.seed_id),
            "seed_to": int(b.seed_id),
            "centroid_displacement_px": float(dist),
            "area_from_px": int(a.segment_area_px),
            "area_to_px": int(b.segment_area_px),
        })

tracks = pd.DataFrame(track_rows)
trans = pd.DataFrame(transition_rows)

tracks.to_csv(
    OUT / "3d_object_topology_qc.csv", index=False
)
trans.to_csv(
    OUT / "3d_adjacent_transition_qc.csv", index=False
)

print("\n=== 3D Object Topology QC ===")
print("Total source IDs:", len(tracks))
print("Multi-section IDs:",
      int((tracks.n_sections >= 2).sum()))
print("IDs with internal gaps:",
      int((tracks.internal_missing_z > 0).sum()))
print("IDs with multiple fragments per section:",
      int((tracks.n_multifrag_sections > 0).sum()))
print("Simple contiguous multi-section candidates:",
      int(tracks.simple_contiguous_candidate.sum()))

print("\n=== Adjacent Geometry ===")
print("Unambiguous 1-to-1 adjacent transitions:", len(trans))

if len(trans):
    print("Centroid displacement quantiles (pixels):")
    print(
        trans.centroid_displacement_px.quantile(
            [0.5, 0.9, 0.95, 0.99]
        ).to_string()
    )

    print("\nLargest centroid displacements:")
    print(
        trans.nlargest(
            10, "centroid_displacement_px"
        ).to_string(index=False)
    )

print("\n=== Saved ===")
print(OUT / "3d_object_topology_qc.csv")
print(OUT / "3d_adjacent_transition_qc.csv")
