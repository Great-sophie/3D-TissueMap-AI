from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "runs/secondher2_3d_object_protein_quantification_v0_1"
)
OUT = Path(
    "runs/qc_secondher2_3d_signal_selection_v0_1"
)
if OUT.exists():
    raise FileExistsError(f"Refusing overwrite: {OUT}")
OUT.mkdir(parents=True)

o = pd.read_csv(ROOT / "3d_her2_object_quantification.csv")
z = pd.read_csv(ROOT / "per_z_her2_intensity_qc.csv")

C1 = "HER2_3B5_mean"
C2 = "HER2_29D8_mean"

needed = {
    "source_pseudo_id", "retained_by_workflow12e",
    "components_6", "n_sections", "n_voxels",
    "touches_xy_roi_edge", C1, C2
}
assert needed.issubset(o.columns)
assert not o.source_pseudo_id.duplicated().any()
assert len(z) == 92 and z.z.nunique() == 92

keep = (
    o.retained_by_workflow12e
    .astype(str).str.lower().eq("true")
)
edge = (
    o.touches_xy_roi_edge
    .astype(str).str.lower().eq("true")
)
geom = (
    keep & (o.components_6 == 1)
    & (o.n_sections >= 2) & ~edge
)

def summarize(name, sub):
    sub = sub.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=[C1, C2])

    return {
        "group": name,
        "n": len(sub),
        "median_voxels": sub.n_voxels.median(),
        "median_sections": sub.n_sections.median(),
        "median_3B5": sub[C1].median(),
        "median_29D8": sub[C2].median(),
        "rho": sub[C1].corr(
            sub[C2], method="spearman"
        ),
        "both_zero_fraction": (
            (sub[C1] == 0) & (sub[C2] == 0)
        ).mean(),
    }

groups = [
    ("all", o),
    ("retained", o[keep]),
    ("excluded", o[~keep]),
    ("geometry_filtered", o[geom]),
]

rows = [summarize(name, sub) for name, sub in groups]

print("\n=== Population selection ===")
print(
    pd.DataFrame(rows).round(4).to_string(index=False)
)

for name, subset in [
    ("retained", o[keep]),
    ("excluded", o[~keep]),
]:
    subset = subset.copy()
    subset["size_quartile"] = pd.qcut(
        subset.n_voxels.rank(method="first"),
        4, labels=False
    )

    for q, part in subset.groupby("size_quartile"):
        rows.append(
            summarize(f"{name}_size_Q{int(q)+1}", part)
        )

results = pd.DataFrame(rows)
results.to_csv(
    OUT / "population_and_size_strata.csv",
    index=False
)

print("\n=== Size-stratified correlations ===")
print(
    results.iloc[4:][
        ["group", "n", "rho", "median_voxels"]
    ].round(4).to_string(index=False)
)

print("\n=== Z-dependent signal checks ===")

for marker in [C1, C2]:
    values = z[marker].replace(
        [np.inf, -np.inf], np.nan
    )
    print(
        marker,
        "Z Spearman:",
        round(z.z.corr(values, method="spearman"), 4),
        "P05:", round(values.quantile(.05), 4),
        "Median:", round(values.median(), 4),
        "P95:", round(values.quantile(.95), 4),
    )

print(
    "Across-Z clone-mean Spearman:",
    z[C1].corr(z[C2], method="spearman")
)

z.to_csv(
    OUT / "per_z_intensity_for_review.csv",
    index=False
)

print("\nSUCCESS:", OUT)
