from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

src = Path("runs/workflow12_secondher2_protein_integration_v0_1")
out = Path("runs/workflow12_secondher2_her2_concordance_v0_1")

if out.exists():
    raise SystemExit(f"Refusing to overwrite: {out}")

d = pd.read_csv(src / "instance_marker_mean_raw.csv")
pz = pd.read_csv(src / "instance_marker_per_z_sums.csv")

a = "mean_raw_HER2_3B5"
b = "mean_raw_HER2_29D8"

assert len(d) == 81226
assert d.instance_id.is_unique
assert sorted(pz.z.unique().tolist()) == list(range(92))
assert not pz.duplicated(["z", "instance_id"]).any()
assert np.isfinite(d[[a, b]].to_numpy()).all()

def rho(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 10 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)

overall = rho(d[a], d[b])
positive = (d[a] > 0) & (d[b] > 0)
positive_rho = rho(d.loc[positive, a], d.loc[positive, b])

n_top = int(np.ceil(len(d) * 0.10))
top_a = set(d.nlargest(n_top, a).instance_id)
top_b = set(d.nlargest(n_top, b).instance_id)
intersection = len(top_a & top_b)
union = len(top_a | top_b)

# Examine correlation in panCK-low and panCK-high strata.
rank = d["mean_raw_panCK"].rank(method="first", pct=True)
strata = {
    "panCK_low25": d[rank <= 0.25],
    "panCK_high25": d[rank > 0.75],
    "multiplane": d[d.n_z_planes >= 2],
}

stratum_rows = [{
    "stratum": "all",
    "n": len(d),
    "spearman": overall
}]

for name, subset in strata.items():
    stratum_rows.append({
        "stratum": name,
        "n": len(subset),
        "spearman": rho(subset[a], subset[b])
    })

# Per-Z signal means are reconstructed using pixel sums
# and each instance's voxel count within that Z plane.
z_rows = []
for z, g in pz.groupby("z"):
    x = g["sum_raw_HER2_3B5"] / g["voxel_count"]
    y = g["sum_raw_HER2_29D8"] / g["voxel_count"]
    z_rows.append({
        "z": int(z),
        "n_section_instances": int(len(g)),
        "spearman": rho(x, y),
        "positive_both_fraction": float(
            ((x > 0) & (y > 0)).mean()
        )
    })

stratum_df = pd.DataFrame(stratum_rows)
z_df = pd.DataFrame(z_rows)

summary = {
    "n_3d_pseudo_instances": int(len(d)),
    "n_z": 92,
    "spearman_all": overall,
    "n_both_positive": int(positive.sum()),
    "spearman_both_positive": positive_rho,
    "fraction_nonzero_3B5": float((d[a] > 0).mean()),
    "fraction_nonzero_29D8": float((d[b] > 0).mean()),
    "top10_n_per_channel": n_top,
    "top10_intersection": intersection,
    "top10_jaccard": intersection / union,
    "median_per_z_spearman": float(
        z_df.spearman.median()
    ),
    "interpretation": (
        "Exploratory antibody-clone agreement in a single "
        "3D tissue volume; not independent cell-boundary accuracy."
    )
}

out.mkdir(parents=True)
stratum_df.to_csv(out / "stratified_clone_correlation.csv", index=False)
z_df.to_csv(out / "per_z_clone_correlation.csv", index=False)
(out / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("===== HER2 CLONE CONCORDANCE =====")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\n===== STRATIFIED CORRELATION =====")
print(stratum_df.to_string(index=False))

print("\n===== PER-Z CORRELATION RANGE =====")
print(z_df.spearman.describe().to_string())
print("\nSAVED:", out)
