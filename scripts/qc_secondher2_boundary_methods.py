from pathlib import Path
import numpy as np
import pandas as pd
import tifffile

root = Path(
    "runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1"
)
out = Path(
    "runs/qc_secondher2_boundary_methods_all92_v0_1"
)
out.mkdir(parents=True, exist_ok=True)

metrics = pd.read_csv(root / "candidate_boundary_metrics.csv")
metrics["testable_interface"] = (
    metrics["testable_interface"]
    .astype(str).str.lower().eq("true")
)

methods = ["cadherin_assisted", "nuclear_voronoi"]
assert set(metrics["method"]) == set(methods)
assert not metrics.duplicated(["method", "z", "seed_id"]).any()

print("\n=== 1. Method and seed audit ===")
print(metrics.groupby("method").agg(
    rows=("seed_id", "size"),
    sections=("z", "nunique"),
    testable_rate=("testable_interface", "mean"),
).to_string())

records = []

for z in range(92):
    folder = root / "candidate_labels"
    a = tifffile.imread(
        folder / f"z{z:03d}_cadherin_assisted.tif"
    )
    b = tifffile.imread(
        folder / f"z{z:03d}_nuclear_voronoi.tif"
    )

    assert a.shape == b.shape

    fa, fb = a > 0, b > 0
    common = fa & fb

    labels_a = np.unique(a[fa])
    labels_b = np.unique(b[fb])
    same_ids = np.array_equal(labels_a, labels_b)

    median_iou = np.nan

    if same_ids and len(labels_a):
        count_a = np.bincount(a.ravel().astype(np.int64))
        count_b = np.bincount(b.ravel().astype(np.int64))

        intersections = np.bincount(
            a[(a == b) & fa].astype(np.int64),
            minlength=max(len(count_a), len(count_b)),
        )

        ids = labels_a.astype(np.int64)
        unions = (
            count_a[ids]
            + count_b[ids]
            - intersections[ids]
        )

        iou = intersections[ids] / np.maximum(unions, 1)
        median_iou = float(np.median(iou))

    records.append({
        "z": z,
        "pixel_identical": bool(np.array_equal(a, b)),
        "foreground_identical": bool(np.array_equal(fa, fb)),
        "same_positive_label_ids": bool(same_ids),
        "n_labels_cadherin": len(labels_a),
        "n_labels_voronoi": len(labels_b),
        "foreground_disagreement_fraction": (
            np.count_nonzero(fa != fb) / a.size
        ),
        "label_disagreement_fraction_common": (
            np.count_nonzero((a != b) & common)
            / max(np.count_nonzero(common), 1)
        ),
        "same_id_median_iou": median_iou,
    })

qc = pd.DataFrame(records)
qc.to_csv(out / "per_z_boundary_method_qc.csv", index=False)

print("\n=== 2. TIFF comparison ===")
print("Exactly identical sections:",
      int(qc.pixel_identical.sum()), "/ 92")
print("Identical foreground sections:",
      int(qc.foreground_identical.sum()), "/ 92")
print("Matching label-ID sets:",
      int(qc.same_positive_label_ids.sum()), "/ 92")

print("\nMedian foreground disagreement:",
      qc.foreground_disagreement_fraction.median())
print("Median label disagreement (common foreground):",
      qc.label_disagreement_fraction_common.median())
print("Median same-ID IoU:",
      qc.same_id_median_iou.median())

print("\n=== 3. Paired measurement comparison ===")

keys = ["z", "seed_id"]

cols = [
    "testable_interface",
    "segment_area_px",
    "n_core_px",
    "n_shared_interface_px",
    "Yb174_Cadherin_interface_minus_core",
    "HER2_3B5_interface_minus_core",
    "HER2_29D8_interface_minus_core",
]

cad = (
    metrics[metrics.method == "cadherin_assisted"]
    .set_index(keys)[cols].add_suffix("_cad")
)
vor = (
    metrics[metrics.method == "nuclear_voronoi"]
    .set_index(keys)[cols].add_suffix("_vor")
)

paired = cad.join(vor, how="inner")
paired.to_csv(out / "paired_boundary_metrics.csv")

print("Cadherin records:", len(cad))
print("Voronoi records:", len(vor))
print("Matched fragments:", len(paired))

valid = (
    paired["testable_interface_cad"]
    & paired["testable_interface_vor"]
)

print("Both methods testable:", int(valid.sum()))

for marker in [
    "Yb174_Cadherin",
    "HER2_3B5",
    "HER2_29D8",
]:
    col = f"{marker}_interface_minus_core"
    x = paired.loc[valid, col + "_cad"]
    y = paired.loc[valid, col + "_vor"]

    good = np.isfinite(x) & np.isfinite(y)
    delta = (x[good] - y[good])

    print(f"\n{marker}")
    print("  Paired valid:", len(delta))

    if len(delta):
        section_medians = delta.groupby(level="z").median()

        print("  Cadherin median:", x[good].median())
        print("  Voronoi median:", y[good].median())
        print("  Paired difference median:", delta.median())
        print("  Section-median difference:",
              section_medians.median())
        print("  Positive difference fraction:",
              (delta > 0).mean())
        print("  Sections evaluated:",
              len(section_medians))

print("\n=== 4. Saved QC outputs ===")
for p in sorted(out.iterdir()):
    print(p)
