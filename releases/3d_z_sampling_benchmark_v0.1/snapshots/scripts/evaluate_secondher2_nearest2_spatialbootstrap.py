#!/usr/bin/env python3
"""SecondHer2: nearest-two-plane control and spatial-block uncertainty.

Reads only previous, frozen tables (no image reprocessing or retuning).
Single held-out middle section per contiguous pseudo-instance; compares nearest
one other plane, nearest two other planes, and all remaining planes.
Block bootstrap is an *internal spatial sensitivity analysis*, not independent
biological replication or a formal guarantee of nominal coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MARKERS = ("HER2_3B5", "HER2_29D8")


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def make_predictions(table: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    required = {"source_pseudo_id", "z", "n_px", *[m + "_sum" for m in MARKERS]}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing per-Z columns: {sorted(missing)}")
    if table.duplicated(["source_pseudo_id", "z"]).any():
        raise ValueError("Duplicate source-ID / Z pair")
    if old.source_pseudo_id.duplicated().any():
        raise ValueError("Duplicate source IDs in heldout predictions")
    if not (table.n_px > 0).all():
        raise ValueError("Nonpositive per-plane pixel count")

    target_ids = set(old.source_pseudo_id.astype(int))
    rows = []
    for sid, g in table[table.source_pseudo_id.isin(target_ids)].groupby("source_pseudo_id", sort=False):
        g = g.sort_values("z").reset_index(drop=True)
        z_values = g.z.to_numpy(dtype=int)
        if len(g) < 4 or not np.all(np.diff(z_values) == 1):
            raise ValueError(f"Expected >=4 contiguous sections for {sid}")
        target_idx = len(g) // 2
        target = g.iloc[target_idx]
        non = g.drop(index=target_idx).copy()
        # Lexicographic stable tie-break: nearest first; preceding Z wins ties.
        nearest_indices = np.lexsort((non.z.to_numpy(), np.abs(non.z.to_numpy() - int(target.z))))
        near = non.iloc[nearest_indices]
        first = near.iloc[:1]
        first_two = near.iloc[:2]
        row = {"source_pseudo_id": int(sid), "n_sections": len(g), "target_z": int(target.z)}
        for marker in MARKERS:
            col = marker + "_sum"
            true_mean = float(target[col] / target.n_px)
            preds = {
                "nearest1": float(first[col].sum() / first.n_px.sum()),
                "nearest2": float(first_two[col].sum() / first_two.n_px.sum()),
                "all_others": float(non[col].sum() / non.n_px.sum()),
            }
            for name, value in preds.items():
                row[f"{marker}_{name}_log_error"] = abs(np.log1p(value) - np.log1p(true_mean))
                row[f"{marker}_{name}_mean"] = value
        rows.append(row)
    pred = pd.DataFrame.from_records(rows)
    if set(pred.source_pseudo_id) != target_ids or len(pred) != len(old):
        raise RuntimeError("Heldout ID mismatch")
    for marker in MARKERS:
        for name, prior in [("nearest1", "single_plane_mean"), ("all_others", "other_planes_mean")]:
            check = pred.set_index("source_pseudo_id")[f"{marker}_{name}_mean"].reindex(old.source_pseudo_id).to_numpy()
            reference = old[f"{marker}_{prior}"].to_numpy()
            if not np.allclose(check, reference, rtol=1e-9, atol=1e-9):
                raise RuntimeError(f"Sanity check failed: {marker} {name} disagrees with frozen evaluation")
    return pred


def grouped_summary(data: pd.DataFrame, group: str, marker: str) -> dict:
    e1 = data[f"{marker}_nearest1_log_error"].to_numpy(float)
    e2 = data[f"{marker}_nearest2_log_error"].to_numpy(float)
    ea = data[f"{marker}_all_others_log_error"].to_numpy(float)
    if not len(e1):
        return {"group": group, "marker": marker, "n_objects": 0}
    return {
        "group": group, "marker": marker, "n_objects": len(data),
        "median_nearest1": float(np.median(e1)),
        "median_nearest2": float(np.median(e2)),
        "median_all_others": float(np.median(ea)),
        "median_paired_reduction_all_vs_1": float(np.median(e1 - ea)),
        "median_paired_reduction_all_vs_2": float(np.median(e2 - ea)),
        "mean_paired_reduction_all_vs_2": float(np.mean(e2 - ea)),
        "fraction_all_better_than_1": float(np.mean(ea < e1)),
        "fraction_all_better_than_2": float(np.mean(ea < e2)),
        "fraction_nearest2_better_than_1": float(np.mean(e2 < e1)),
    }


def block_bootstrap_primary(data: pd.DataFrame, marker: str, rng: np.random.Generator, B: int):
    diff = (data[f"{marker}_nearest2_log_error"] - data[f"{marker}_all_others_log_error"]).to_numpy(float)
    # Primary estimated effect: average paired *absolute-log-error* reduction.
    # Resample spatial tiles, keeping all included objects jointly.
    group_codes, block_names = pd.factorize(data["spatial_block"], sort=True)
    n_blocks = len(block_names)
    if n_blocks < 15:
        raise ValueError(f"Too few populated spatial blocks: {n_blocks}")
    obj_idx_by_block = [np.flatnonzero(group_codes == i) for i in range(n_blocks)]
    estimates = np.empty(B, dtype=float)
    for i in range(B):
        selection = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([obj_idx_by_block[k] for k in selection])
        estimates[i] = np.mean(diff[idx])
    return {
        "marker": marker,
        "n_objects": len(diff),
        "n_spatial_blocks": n_blocks,
        "mean_paired_reduction_all_vs_nearest2": float(np.mean(diff)),
        "block_bootstrap_p025": float(np.quantile(estimates, 0.025)),
        "block_bootstrap_p975": float(np.quantile(estimates, 0.975)),
        "bootstrap_repetitions": B,
        "interpretation": "within-sample spatial block resampling; not independent biological replication",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("runs/secondher2_nearest2_spatial_robustness_v0_1"))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--block-xy-um", type=float, default=32.)
    ap.add_argument("--block-z-um", type=float, default=24.)
    args = ap.parse_args()
    if args.bootstrap < 100 or args.block_xy_um <= 0 or args.block_z_um <= 0:
        ap.error("Bootstrap must be >=100 and block dimensions >0")
    base = args.project_root.expanduser().resolve()
    held = base / "runs/secondher2_2d_vs_3d_heldout_v0_1"
    coords = base / "runs/secondher2_3d_spatial_discordance_v0_1/3d_her2_object_spatial_coordinates.csv"
    out = base / args.out
    if out.exists():
        raise FileExistsError(f"Refusing overwrite: {out}")
    for p in (held / "per_object_per_section_her2.csv", held / "heldout_2d_vs_multiplane_predictions.csv", coords):
        if not p.is_file():
            raise FileNotFoundError(f"Required previous output missing: {p}")

    per_z = pd.read_csv(held / "per_object_per_section_her2.csv")
    old = pd.read_csv(held / "heldout_2d_vs_multiplane_predictions.csv")
    pred = make_predictions(per_z, old)
    meta = old[["source_pseudo_id", "primary_eligible", "retained", "n_sections"]]
    pred = pred.merge(meta, on="source_pseudo_id", suffixes=("", "_previous"), validate="one_to_one")
    if (pred.n_sections != pred.n_sections_previous).any():
        raise RuntimeError("n_sections mismatch")
    pred = pred.drop(columns="n_sections_previous")
    coords_tbl = pd.read_csv(coords)
    if not coords_tbl.source_pseudo_id.is_unique:
        raise ValueError("Duplicate coordinate source IDs")
    pred = pred.merge(coords_tbl[["source_pseudo_id", "x_um", "y_um", "z_um"]],
                      on="source_pseudo_id", validate="one_to_one", how="left")
    if pred[["x_um", "y_um", "z_um"]].isna().any().any():
        raise ValueError("Missing spatial object coordinates")

    bx = np.floor(pred.x_um / args.block_xy_um).astype(int)
    by = np.floor(pred.y_um / args.block_xy_um).astype(int)
    bz = np.floor(pred.z_um / args.block_z_um).astype(int)
    pred["spatial_block"] = bx.astype(str) + ":" + by.astype(str) + ":" + bz.astype(str)
    group_sets = {
        "primary_retained_clean": pred.loc[as_bool(pred.primary_eligible)],
        "all_retained": pred.loc[as_bool(pred.retained)],
        "all_excluded": pred.loc[~as_bool(pred.retained)],
    }
    summary = []
    for label, sub in group_sets.items():
        for marker in MARKERS:
            summary.append(grouped_summary(sub, label, marker))
    for label, sub in [("primary_z4", group_sets["primary_retained_clean"].loc[lambda x: x.n_sections == 4]),
                       ("primary_z5_6", group_sets["primary_retained_clean"].loc[lambda x: x.n_sections.between(5, 6)]),
                       ("primary_z7plus", group_sets["primary_retained_clean"].loc[lambda x: x.n_sections >= 7])]:
        for marker in MARKERS:
            summary.append(grouped_summary(sub, label, marker))
    summary = pd.DataFrame(summary)
    rng = np.random.default_rng(20261020)
    boots = [block_bootstrap_primary(group_sets["primary_retained_clean"], marker, rng, args.bootstrap)
             for marker in MARKERS]
    boot = pd.DataFrame(boots)

    out.mkdir(parents=True)
    pred.to_csv(out / "object_nearest1_nearest2_all_predictions.csv", index=False)
    summary.to_csv(out / "nearest2_control_summary.csv", index=False)
    boot.to_csv(out / "spatial_block_bootstrap_primary.csv", index=False)
    (out / "protocol.json").write_text(json.dumps({
        "target": "one middle Z plane per contiguous pseudo-instance, unchanged from prior evaluation",
        "metric": "absolute log1p(raw mean intensity) prediction error",
        "predictors": {"nearest1": "voxel mean of nearest non-target Z plane", "nearest2": "voxel-weighted mean of two nearest non-target Z planes", "all_others": "voxel-weighted mean of all non-target Z planes"},
        "bootstrap": "resample spatial blocks (32x32x24 um default) with replacement in primary subset",
        "caveat": "single specimen, author-produced pseudo-instances, variable information budget across methods, descriptive internal spatial confidence bands",
        "bootstrap_repetitions": args.bootstrap,
        "block_xy_um": args.block_xy_um,
        "block_z_um": args.block_z_um,
    }, indent=2, ensure_ascii=False))
    print("\n=== NEAREST-TWO-PLANE CONTROL ===")
    print(summary[["group", "marker", "n_objects", "median_nearest1", "median_nearest2", "median_all_others", "median_paired_reduction_all_vs_2", "fraction_all_better_than_2"]].round(5).to_string(index=False))
    print("\n=== PRIMARY SPATIAL-BLOCK BOOTSTRAP (MEAN paired reduction: nearest2 - all) ===")
    print(boot[["marker", "n_objects", "n_spatial_blocks", "mean_paired_reduction_all_vs_nearest2", "block_bootstrap_p025", "block_bootstrap_p975"]].round(6).to_string(index=False))
    print("\nSUCCESS:", out)


if __name__ == "__main__":
    main()
