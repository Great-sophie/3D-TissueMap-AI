from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1"
)
OUT = Path(
    "runs/qc_secondher2_boundary_methods_all92_v0_1"
)
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "candidate_boundary_metrics.csv")

keys = ["z", "seed_id"]
cols = [
    "testable_interface",
    "HER2_3B5_interface_minus_core",
    "HER2_29D8_interface_minus_core",
]

assert not df.duplicated(["method", *keys]).any()

cad = (
    df[df.method == "cadherin_assisted"]
    .set_index(keys)[cols]
    .add_suffix("_cad")
)

vor = (
    df[df.method == "nuclear_voronoi"]
    .set_index(keys)[cols]
    .add_suffix("_vor")
)

p = cad.join(vor, how="inner", validate="one_to_one")

for method in ["cad", "vor"]:
    col = f"testable_interface_{method}"
    p[col] = p[col].astype(str).str.lower().eq("true")

records = []

def spearman(x, y):
    if len(x) < 10:
        return np.nan
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return x.corr(y, method="spearman")

for z, g in p.groupby(level="z"):
    eligible = (
        g.testable_interface_cad
        & g.testable_interface_vor
    )

    q = g.loc[eligible].copy()

    needed = [
        f"HER2_{clone}_interface_minus_core_{method}"
        for clone in ["3B5", "29D8"]
        for method in ["cad", "vor"]
    ]

    q = q[needed].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    correlations = {}

    for method in ["cad", "vor"]:
        x = q[
            f"HER2_3B5_interface_minus_core_{method}"
        ]
        y = q[
            f"HER2_29D8_interface_minus_core_{method}"
        ]
        correlations[method] = spearman(x, y)

    records.append({
        "z": z,
        "n_fragments": len(g),
        "n_testable_cad": int(g.testable_interface_cad.sum()),
        "n_testable_vor": int(g.testable_interface_vor.sum()),
        "n_both_testable": int(eligible.sum()),
        "n_valid_dualclone": len(q),
        "spearman_cad": correlations["cad"],
        "spearman_vor": correlations["vor"],
        "delta_spearman": (
            correlations["cad"] - correlations["vor"]
        ),
    })

result = pd.DataFrame(records)
path = OUT / "per_z_dualclone_concordance.csv"
result.to_csv(path, index=False)

valid = result.dropna(subset=["delta_spearman"])

print("\n=== Dual-clone Concordance ===")
print("Sections audited:", len(result))
print("Sections with evaluable correlations:", len(valid))
print("Total both-testable fragments:",
      result.n_both_testable.sum())

print("\nMedian per-section Spearman:")
print("Cadherin-assisted:", valid.spearman_cad.median())
print("Nuclear Voronoi:", valid.spearman_vor.median())
print("Paired rho difference:", valid.delta_spearman.median())

print("\nSections with higher concordance:")
print("Cadherin-assisted:",
      int((valid.delta_spearman > 0).sum()))
print("Nuclear Voronoi:",
      int((valid.delta_spearman < 0).sum()))
print("Ties:", int((valid.delta_spearman == 0).sum()))

print("\n=== Lowest delta sections ===")
print(
    valid.nsmallest(5, "delta_spearman")
    [["z", "n_valid_dualclone", "delta_spearman"]]
    .to_string(index=False)
)

print("\n=== Highest delta sections ===")
print(
    valid.nlargest(5, "delta_spearman")
    [["z", "n_valid_dualclone", "delta_spearman"]]
    .to_string(index=False)
)

print("\nSaved:", path)
