from pathlib import Path
import pandas as pd

root = Path(
    "runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1"
)
out = Path(
    "runs/qc_secondher2_boundary_methods_all92_v0_1"
)
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(root / "candidate_boundary_metrics.csv")

# Avoid counting the same fragment twice across methods.
d = df[df["method"] == "nuclear_voronoi"].copy()

assert len(d) == 11716
assert not d.duplicated(["z", "seed_id"]).any()

print("\n=== 1. Source pseudo-ID audit ===")
print("Total nuclear fragments:", len(d))
print("Missing source IDs:", d.source_pseudo_id.isna().sum())
print("Unique source IDs:", d.source_pseudo_id.nunique())

tracks = (
    d.dropna(subset=["source_pseudo_id"])
    .groupby("source_pseudo_id")
    .agg(
        n_fragments=("seed_id", "size"),
        n_sections=("z", "nunique"),
        z_min=("z", "min"),
        z_max=("z", "max"),
    )
)

tracks["z_span"] = tracks.z_max - tracks.z_min + 1
tracks["internal_missing_z"] = (
    tracks.z_span - tracks.n_sections
)

print("\n=== 2. Cross-section identity ===")
print("IDs appearing in one section:",
      int((tracks.n_sections == 1).sum()))
print("IDs appearing in multiple sections:",
      int((tracks.n_sections > 1).sum()))

print("IDs spanning >= 3 sections:",
      int((tracks.n_sections >= 3).sum()))

print("Maximum sections per ID:",
      int(tracks.n_sections.max()))

print("Median sections per ID:",
      float(tracks.n_sections.median()))

print("\n=== 3. Continuity ===")
multi = tracks[tracks.n_sections > 1]

print("Multi-section IDs without internal gaps:",
      int((multi.internal_missing_z == 0).sum()))

print("Multi-section IDs with internal gaps:",
      int((multi.internal_missing_z > 0).sum()))

duplicates = (
    d.groupby(["source_pseudo_id", "z"])
    .size()
)

print("ID-section pairs with multiple fragments:",
      int((duplicates > 1).sum()))

print("\n=== 4. Most extensive IDs ===")
print(
    tracks.sort_values(
        ["n_sections", "n_fragments"],
        ascending=False
    ).head(10).to_string()
)

path = out / "source_pseudo_id_continuity_audit.csv"
tracks.to_csv(path)

print("\nSaved:", path)
