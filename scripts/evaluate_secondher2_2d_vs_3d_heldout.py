#!/usr/bin/env python3
"""SecondHer2: held-out-plane prediction with 2D versus multi-plane evidence.

Exploratory within-specimen assessment. Original author watershed pseudo-instances
are not ground-truth cells. 2D and multi-plane predictors never use target pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import xml.etree.ElementTree as ET


def strbool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().eq("true")


def collect_section_stats(ome: Path, mask: Path, summary: dict) -> pd.DataFrame:
    x0, x1, y0, y1 = map(int, summary["roi_xy"])
    zs = list(map(int, summary["z_indices"]))
    ch = summary["channel_index"]
    expected = {"HER2_3B5": "Eu151", "HER2_29D8": "Gd156"}
    recs = []
    with tifffile.TiffFile(ome) as tf, tifffile.TiffFile(mask) as mf:
        s = tf.series[0]
        if s.axes != "ZCYX" or s.shape != (92, 31, 686, 608):
            raise ValueError(f"Unexpected OME data geometry {s.axes} {s.shape}")
        if len(mf.pages) != 92:
            raise ValueError("Expected 92 mask pages")
        root = ET.fromstring(tf.ome_metadata)
        px = next(x for x in root.iter() if x.tag.endswith("Pixels"))
        names = [c.attrib.get("Name") for c in px if c.tag.endswith("Channel")]
        for marker, metal in expected.items():
            if names[int(ch[marker])] != metal:
                raise ValueError(f"Channel mismatch for {marker}: {names[int(ch[marker])]}")
        for z in zs:
            labels = mf.pages[z].asarray()[y0:y1, x0:x1]
            if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
                raise ValueError(f"Invalid labels at z={z}")
            fg = labels > 0
            if not fg.any():
                continue
            ids, inv, counts = np.unique(labels[fg], return_inverse=True, return_counts=True)
            sums = {}
            for marker in expected:
                im = s.pages[z * s.shape[1] + int(ch[marker])].asarray()[y0:y1, x0:x1]
                if not np.all(np.isfinite(im)) or np.any(im < 0):
                    raise ValueError(f"Invalid intensity at z={z}, marker={marker}")
                sums[marker] = np.bincount(inv, weights=im[fg].astype(float), minlength=len(ids))
            for i, sid in enumerate(ids):
                recs.append({
                    "source_pseudo_id": int(sid), "z": int(z),
                    "n_px": int(counts[i]),
                    "HER2_3B5_sum": float(sums["HER2_3B5"][i]),
                    "HER2_29D8_sum": float(sums["HER2_29D8"][i]),
                })
            if z % 10 == 0 or z == zs[-1]:
                print(f"Processed z={z}/{zs[-1]}", flush=True)
    table = pd.DataFrame.from_records(recs)
    if table.duplicated(["source_pseudo_id", "z"]).any():
        raise RuntimeError("Duplicate source ID per section")
    return table


def make_heldout_comparison(table: pd.DataFrame, obj: pd.DataFrame, min_sections: int) -> pd.DataFrame:
    records = []
    for sid, g in table.groupby("source_pseudo_id", sort=True):
        if len(g) < min_sections:
            continue
        g = g.sort_values("z").reset_index(drop=True)
        zs = g["z"].to_numpy(dtype=int)
        if not np.all(np.diff(zs) == 1):
            continue  # No interpolation across missing sections
        target_idx = len(g) // 2
        target = g.iloc[target_idx]
        train = g.drop(index=target_idx)
        # Fairly strong 2D baseline: nearest non-held-out physical section.
        nearest_idx = int(np.argmin(np.abs(train.z.to_numpy() - target.z)))
        nearest = train.iloc[nearest_idx]
        row = {
            "source_pseudo_id": int(sid),
            "n_sections": len(g),
            "target_z": int(target.z),
            "baseline_z": int(nearest.z),
            "n_train_sections": len(train),
            "target_n_px": int(target.n_px),
            "baseline_n_px": int(nearest.n_px),
            "train_n_px": int(train.n_px.sum()),
        }
        for m in ["HER2_3B5", "HER2_29D8"]:
            k = m + "_sum"
            target_mean = float(target[k] / target.n_px)
            single_mean = float(nearest[k] / nearest.n_px)
            multi_mean = float(train[k].sum() / train.n_px.sum())
            err_single = abs(np.log1p(single_mean) - np.log1p(target_mean))
            err_multi = abs(np.log1p(multi_mean) - np.log1p(target_mean))
            row.update({
                f"{m}_heldout_mean": target_mean,
                f"{m}_single_plane_mean": single_mean,
                f"{m}_other_planes_mean": multi_mean,
                f"{m}_single_log_error": err_single,
                f"{m}_multiplane_log_error": err_multi,
                f"{m}_error_reduction": err_single - err_multi,
            })
        records.append(row)
    out = pd.DataFrame.from_records(records)
    if out.empty:
        raise RuntimeError("No eligible contiguous multi-section objects")
    return out.merge(obj, on="source_pseudo_id", validate="one_to_one")


def run(args: argparse.Namespace) -> None:
    root = args.project_root.expanduser().resolve()
    wf_dir = root / "runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1"
    quant_file = root / "runs/secondher2_3d_object_protein_quantification_v0_1/3d_her2_object_quantification.csv"
    summary = json.loads((wf_dir / "summary.json").read_text())
    base = root / "data/inspect/SecondHer2BreastCancerModel/SecondHer2BreastCancerModel"
    ome = base / "aligned_SecondHer2BreastCancer_model_3D_image_stack.ome.tiff"
    mask = base / "measured_mask_final_segmentation_hwatershed_bg500_90%.tif"
    out_dir = root / args.out
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {out_dir}")
    objects = pd.read_csv(quant_file)
    if objects.source_pseudo_id.duplicated().any():
        raise ValueError("Duplicate object IDs in quantification")
    print("Building per-object per-Z HER2 table from unchanged source images...")
    table = collect_section_stats(ome, mask, summary)
    counts = table.groupby("source_pseudo_id").n_px.sum()
    original = objects.set_index("source_pseudo_id")
    if set(counts.index) != set(original.index):
        raise RuntimeError("Object-ID coverage differs from previous quantification")
    if not np.array_equal(counts.sort_index().to_numpy(), original.loc[counts.sort_index().index, "n_voxels"].to_numpy()):
        raise RuntimeError("Voxel totals disagree with original 3D quantification")
    for m in ["HER2_3B5", "HER2_29D8"]:
        sums = table.groupby("source_pseudo_id")[m + "_sum"].sum()
        means = sums / counts
        earlier = original.loc[means.index, m + "_mean"]
        if not np.allclose(means.to_numpy(), earlier.to_numpy(), atol=1e-6, rtol=1e-5):
            raise RuntimeError(f"{m} per-Z aggregation fails previous 3D mean validation")

    cols = ["source_pseudo_id", "retained_by_workflow12e", "components_6", "touches_xy_roi_edge", "touches_z_stack_edge", "n_voxels"]
    comparisons = make_heldout_comparison(table, objects[cols], args.min_sections)
    comparisons["retained"] = strbool(comparisons.retained_by_workflow12e)
    comparisons["touches_xy"] = strbool(comparisons.touches_xy_roi_edge)
    comparisons["touches_stack"] = strbool(comparisons.touches_z_stack_edge)
    comparisons["primary_eligible"] = (
        comparisons.retained & (comparisons.components_6 == 1)
        & ~comparisons.touches_xy & ~comparisons.touches_stack
    )
    print("\n=== HELD-OUT SECTION VALIDATION ===")
    print("Eligible contiguous pseudo-instances with >=", args.min_sections, "Z sections:", len(comparisons))
    print("Primary retained, 6-connected, non-edge subset:", int(comparisons.primary_eligible.sum()))
    lines = []
    for name, subset in [
        ("primary_retained_clean", comparisons[comparisons.primary_eligible]),
        ("all_retained", comparisons[comparisons.retained]),
        ("all_excluded", comparisons[~comparisons.retained]),
    ]:
        for m in ["HER2_3B5", "HER2_29D8"]:
            d2 = subset[f"{m}_single_log_error"].to_numpy()
            d3 = subset[f"{m}_multiplane_log_error"].to_numpy()
            reduction = d2 - d3
            row = {
                "group": name, "marker": m, "n_objects": len(subset),
                "median_single_log_error": float(np.median(d2)) if len(d2) else np.nan,
                "median_multiplane_log_error": float(np.median(d3)) if len(d3) else np.nan,
                "median_paired_error_reduction": float(np.median(reduction)) if len(reduction) else np.nan,
                "fraction_multiplane_better": float(np.mean(reduction > 0)) if len(reduction) else np.nan,
                "fraction_ties": float(np.mean(np.isclose(reduction, 0, atol=1e-12))) if len(reduction) else np.nan,
            }
            lines.append(row)
    summary_table = pd.DataFrame(lines)
    print(summary_table.round(5).to_string(index=False))

    out_dir.mkdir(parents=True)
    table.to_csv(out_dir / "per_object_per_section_her2.csv", index=False)
    comparisons.to_csv(out_dir / "heldout_2d_vs_multiplane_predictions.csv", index=False)
    summary_table.to_csv(out_dir / "heldout_comparison_summary.csv", index=False)
    (out_dir / "protocol.json").write_text(json.dumps({
        "task": "same-pseudo-instance held-out-middle-plane prediction",
        "target": "middle occupied plane of each contiguous source pseudo-instance",
        "single_plane_baseline": "nearest other Z plane; tie broken in favor of preceding Z",
        "multi_plane_predictor": "voxel-weighted average of ALL non-target Z planes",
        "metric": "absolute log1p(predicted raw mean) minus log1p(held-out raw mean)",
        "minimum_sections": args.min_sections,
        "interpretation": "internal slice reproducibility, NOT ground-truth cell accuracy nor independent replication",
        "source": str(ome), "mask": str(mask), "roi_xy": summary["roi_xy"],
        "z_indices": summary["z_indices"],
    }, ensure_ascii=False, indent=2))
    print("\nSUCCESS:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--min-sections", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("runs/secondher2_2d_vs_3d_heldout_v0_1"))
    arg = ap.parse_args()
    if arg.min_sections < 3:
        ap.error("--min-sections must be at least 3")
    run(arg)
