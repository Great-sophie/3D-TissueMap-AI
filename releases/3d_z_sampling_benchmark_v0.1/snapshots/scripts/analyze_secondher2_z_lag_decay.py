#!/usr/bin/env python3
"""Descriptive HER2 Z-lag profile using frozen per-object/per-Z measurements.

One pre-selected held-out middle Z per pseudo-instance. Compare single plane
measurements at distance |delta Z|=1,2,... against that target without tuning.
Not a biological replication or independent validation study.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

MARKERS = ("HER2_3B5", "HER2_29D8")

def bool_series(x):
    return x.astype(str).str.lower().eq("true")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("runs/secondher2_z_lag_profile_v0_1"))
    ap.add_argument("--max-lag", type=int, default=8)
    args = ap.parse_args()
    if args.max_lag < 2:
        ap.error("--max-lag must be >=2")
    root = args.project_root.resolve()
    src = root / "runs/secondher2_2d_vs_3d_heldout_v0_1"
    path_table = src / "per_object_per_section_her2.csv"
    path_old = src / "heldout_2d_vs_multiplane_predictions.csv"
    out = root / args.out
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    ztab = pd.read_csv(path_table)
    old = pd.read_csv(path_old)
    req_z = {"source_pseudo_id", "z", "n_px", *[m + "_sum" for m in MARKERS]}
    req_old = {"source_pseudo_id", "target_z", "retained", "primary_eligible",
               *[m + "_heldout_mean" for m in MARKERS]}
    if req_z - set(ztab) or req_old - set(old):
        raise ValueError(f"Missing columns: perZ={req_z - set(ztab)}, heldout={req_old - set(old)}")
    if ztab.duplicated(["source_pseudo_id", "z"]).any() or old.source_pseudo_id.duplicated().any():
        raise ValueError("Duplicate ID keys")
    if not (ztab.n_px > 0).all():
        raise ValueError("Invalid voxel count")
    if not old.source_pseudo_id.isin(ztab.source_pseudo_id).all():
        raise ValueError("Missing held-out IDs in per-Z measurements")
    old = old.copy()
    old["primary"] = bool_series(old.primary_eligible)
    old["kept"] = bool_series(old.retained)
    merged = ztab.merge(old[["source_pseudo_id", "target_z", "primary", "kept", *[m + "_heldout_mean" for m in MARKERS]]],
                        on="source_pseudo_id", how="inner", validate="many_to_one")
    merged["lag"] = (merged.z - merged.target_z).abs().astype(int)
    for m in MARKERS:
        obs = merged[m + "_sum"] / merged.n_px
        if (obs < 0).any():
            raise ValueError("Negative intensity in " + m)
        actual = merged[m + "_heldout_mean"]
        merged[m + "_logerr"] = np.abs(np.log1p(obs) - np.log1p(actual))
    target = merged.loc[merged.lag == 0]
    if len(target) != len(old) or target.source_pseudo_id.nunique() != len(old):
        raise ValueError("Expected exactly one target plane for each held-out object")
    for m in MARKERS:
        v = target[m + "_sum"] / target.n_px
        if not np.allclose(v, target[m + "_heldout_mean"], atol=1e-8, rtol=1e-8):
            raise RuntimeError("Frozen held-out targets fail reconciliation for " + m)
    merged = merged[(merged.lag >= 1) & (merged.lag <= args.max_lag)]
    frames = []
    for m in MARKERS:
        temp = merged.groupby(["source_pseudo_id", "lag"], as_index=False).agg(
            mean_abs_log_error=(m + "_logerr", "mean"),
            n_planes=("z", "size"),
            kept=("kept", "first"), primary=("primary", "first"),
        )
        temp["marker"] = m
        frames.append(temp)
    obj = pd.concat(frames, ignore_index=True)
    obj["lag_um"] = obj.lag * 2.0  # From frozen OME PhysicalSizeZ = 2 µm
    reference = obj.loc[obj.lag == 1, ["source_pseudo_id", "marker", "mean_abs_log_error"]]
    reference = reference.rename(columns={"mean_abs_log_error": "lag1_error"})
    obj = obj.merge(reference, on=["source_pseudo_id", "marker"], how="left", validate="many_to_one")
    obj["extra_error_vs_lag1"] = obj.mean_abs_log_error - obj.lag1_error
    rows = []
    groups = {
        "primary_retained_clean": obj.loc[obj.primary],
        "all_retained": obj.loc[obj.kept],
        "all_excluded": obj.loc[~obj.kept],
    }
    for label, g in groups.items():
        for (marker, lag), part in g.groupby(["marker", "lag"]):
            matched = part.dropna(subset=["lag1_error"])
            rows.append({
                "group": label, "marker": marker, "z_lag": int(lag), "physical_lag_um": float(lag * 2),
                "n_objects": len(part), "median_single_plane_error": float(part.mean_abs_log_error.median()),
                "n_paired_lag1": len(matched),
                "median_paired_extra_error_vs_lag1": float(matched.extra_error_vs_lag1.median()) if len(matched) else np.nan,
                "fraction_more_distant_worse_than_lag1": float((matched.extra_error_vs_lag1 > 0).mean()) if len(matched) else np.nan,
            })
    summary = pd.DataFrame(rows).sort_values(["group", "marker", "z_lag"])
    out.mkdir(parents=True)
    obj.to_csv(out / "per_object_z_lag_error.csv", index=False)
    summary.to_csv(out / "z_lag_error_summary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "target": "identical preselected middle section as frozen held-out experiment",
        "metric": "mean per-object of absolute log1p intensity error across single source planes at same absolute Z lag",
        "comparison": "within-object paired lag-k versus lag-1 error (if lag-k exists)",
        "physical_size_z_um": 2.0,
        "max_lag": args.max_lag,
        "caveats": ["single specimen", "pseudo-instance IDs are not independently proven cells", "longer lags involve fewer/longer objects", "descriptive statistics only", "OME nominal spacing"],
    }, indent=2), encoding="utf-8")
    print("\n=== Z-LAG PROFILE (held-out middle Z) ===")
    print(summary.loc[(summary.group == "primary_retained_clean") & (summary.z_lag <= 5)].round(5).to_string(index=False))
    print("\n=== EXCLUDED COMPARISON ===")
    print(summary.loc[(summary.group == "all_excluded") & (summary.z_lag <= 5)].round(5).to_string(index=False))
    print("\nSUCCESS:", out)

if __name__ == "__main__":
    main()
