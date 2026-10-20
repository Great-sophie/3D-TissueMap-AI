from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

ROOT = Path(
    "runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1"
)
QC = Path(
    "runs/qc_secondher2_boundary_methods_all92_v0_1"
)
QC.mkdir(parents=True, exist_ok=True)

trans = pd.read_csv(
    QC / "3d_adjacent_transition_qc.csv"
)
topo = pd.read_csv(
    QC / "3d_object_topology_qc.csv"
)

assert len(trans) > 0
assert (trans.z_to == trans.z_from + 1).all()
assert not trans.duplicated(
    ["source_pseudo_id", "z_from", "z_to"]
).any()

METHODS = ["nuclear_voronoi", "cadherin_assisted"]


def overlap_stats(a, b, boxes_a, boxes_b, ida, idb):
    ida, idb = int(ida), int(idb)

    if ida < 1 or idb < 1:
        raise ValueError("Seed labels must be positive")

    if ida > len(boxes_a) or idb > len(boxes_b):
        raise ValueError("Seed label outside TIFF range")

    sa = boxes_a[ida - 1]
    sb = boxes_b[idb - 1]

    if sa is None or sb is None:
        raise ValueError("Seed label missing from TIFF")

    area_a = int(np.count_nonzero(a[sa] == ida))
    area_b = int(np.count_nonzero(b[sb] == idb))

    y0 = max(sa[0].start, sb[0].start)
    y1 = min(sa[0].stop, sb[0].stop)
    x0 = max(sa[1].start, sb[1].start)
    x1 = min(sa[1].stop, sb[1].stop)

    intersection = 0

    if y1 > y0 and x1 > x0:
        aa = a[y0:y1, x0:x1] == ida
        bb = b[y0:y1, x0:x1] == idb
        intersection = int(np.count_nonzero(aa & bb))

    union = area_a + area_b - intersection

    return {
        "overlap_px": intersection,
        "iou": intersection / union if union else np.nan,
        "dice": (
            2 * intersection / (area_a + area_b)
            if (area_a + area_b) else np.nan
        ),
        "overlap_smaller": (
            intersection / min(area_a, area_b)
            if min(area_a, area_b) else np.nan
        ),
        "area_ratio": (
            max(area_a, area_b) / min(area_a, area_b)
            if min(area_a, area_b) else np.nan
        ),
        "area_from": area_a,
        "area_to": area_b,
    }


rows = []

for z, g in trans.groupby("z_from", sort=True):
    arrays = {}
    boxes = {}

    for method in METHODS:
        a = tifffile.imread(
            ROOT / "candidate_labels" /
            f"z{z:03d}_{method}.tif"
        )
        b = tifffile.imread(
            ROOT / "candidate_labels" /
            f"z{z+1:03d}_{method}.tif"
        )

        assert a.shape == b.shape

        arrays[method] = (a, b)
        boxes[method] = (
            ndi.find_objects(a),
            ndi.find_objects(b)
        )

    for r in g.itertuples(index=False):
        row = {
            "source_pseudo_id": int(r.source_pseudo_id),
            "z_from": int(r.z_from),
            "z_to": int(r.z_to),
            "seed_from": int(r.seed_from),
            "seed_to": int(r.seed_to),
            "centroid_displacement_px": float(
                r.centroid_displacement_px
            ),
        }

        for method in METHODS:
            a, b = arrays[method]
            ba, bb = boxes[method]

            stats = overlap_stats(
                a, b, ba, bb,
                r.seed_from, r.seed_to
            )

            prefix = (
                "vor" if method == "nuclear_voronoi"
                else "cad"
            )

            for k, v in stats.items():
                row[f"{prefix}_{k}"] = v

        assert row["vor_area_from"] == int(r.area_from_px)
        assert row["vor_area_to"] == int(r.area_to_px)

        rows.append(row)

result = pd.DataFrame(rows)

result["delta_dice"] = (
    result.cad_dice - result.vor_dice
)

# Empirical screening flags, not biological failure rules.
p95 = result.centroid_displacement_px.quantile(0.95)

result["flag_large_displacement"] = (
    result.centroid_displacement_px > p95
)
result["flag_zero_overlap"] = (
    result.vor_overlap_px == 0
)
result["flag_area_ratio_gt4"] = (
    result.vor_area_ratio > 4
)

result["flag_review"] = result[
    [
        "flag_large_displacement",
        "flag_zero_overlap",
        "flag_area_ratio_gt4"
    ]
].any(axis=1)

result.to_csv(
    QC / "3d_adjacent_spatial_overlap_qc.csv",
    index=False
)

print("\n=== Adjacent Spatial Overlap QC ===")
print("Transitions evaluated:", len(result))

for prefix, name in [
    ("vor", "Nuclear Voronoi"),
    ("cad", "Cadherin-assisted")
]:
    print(f"\n{name}")
    print("Zero overlap:",
          int((result[f"{prefix}_overlap_px"] == 0).sum()))
    print("Median Dice:",
          result[f"{prefix}_dice"].median())
    print("Median IoU:",
          result[f"{prefix}_iou"].median())
    print("Median smaller-region overlap:",
          result[f"{prefix}_overlap_smaller"].median())

print("\n=== Method Sensitivity ===")
print("Median paired Dice difference (Cad - Vor):",
      result.delta_dice.median())
print("Transitions with higher Cadherin Dice:",
      int((result.delta_dice > 0).sum()))
print("Transitions with higher Voronoi Dice:",
      int((result.delta_dice < 0).sum()))
print("Ties:",
      int((result.delta_dice == 0).sum()))

print("\n=== Screening Flags ===")
print("Displacement > empirical P95:",
      int(result.flag_large_displacement.sum()))
print("Zero Voronoi overlap:",
      int(result.flag_zero_overlap.sum()))
print("Voronoi area ratio > 4:",
      int(result.flag_area_ratio_gt4.sum()))
print("Transitions flagged for review:",
      int(result.flag_review.sum()))

# Evaluate overlap continuity among simple-topology IDs.
simple = topo.loc[
    topo.simple_contiguous_candidate
        .astype(str).str.lower().eq("true"),
    "source_pseudo_id"
]

simple_edges = result[
    result.source_pseudo_id.isin(simple)
]

edge_summary = simple_edges.groupby(
    "source_pseudo_id"
).agg(
    n_edges=("vor_dice", "size"),
    all_positive_overlap=(
        "vor_overlap_px",
        lambda x: bool((x > 0).all())
    ),
    min_dice=("vor_dice", "min")
)

print("\n=== Simple-topology Candidates ===")
print("Candidates:", len(simple))
print("Candidates with evaluated edges:",
      len(edge_summary))
print("Candidates with positive overlap on all edges:",
      int(edge_summary.all_positive_overlap.sum()))
print("Median minimum adjacent Dice:",
      edge_summary.min_dice.median())

edge_summary.to_csv(
    QC / "3d_simple_candidate_overlap_summary.csv"
)

print("\n=== Highest Priority Review Cases ===")
cols = [
    "source_pseudo_id", "z_from", "z_to",
    "centroid_displacement_px",
    "vor_dice", "cad_dice", "vor_area_ratio"
]

print(
    result[result.flag_review]
    .sort_values(
        "centroid_displacement_px",
        ascending=False
    )[cols]
    .head(12)
    .to_string(index=False)
)

print("\nSaved QC results in:", QC)
