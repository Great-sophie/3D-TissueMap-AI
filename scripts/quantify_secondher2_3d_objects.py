#!/usr/bin/env python3
"""Exploratory 3D HER2 measurements on author-provided SecondHer2 pseudo-instance labels.

Reads source inputs without modification. Treats labels as candidate regions, NOT
validated single-cell boundaries or independent biological replicates.

Run from any working directory after placing this file under PROJECT/scripts/:
    python scripts/quantify_secondher2_3d_objects.py
"""

import argparse
import json
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import tifffile


PROJECT = Path(__file__).resolve().parent.parent
RUN_REL = "runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1"
QC_REL = "runs/qc_secondher2_boundary_methods_all92_v0_1"
OUT_REL = "runs/secondher2_3d_object_protein_quantification_v0_1"
BASE_REL = "data/inspect/SecondHer2BreastCancerModel/SecondHer2BreastCancerModel"
OME_NAME = "aligned_SecondHer2BreastCancer_model_3D_image_stack.ome.tiff"
MASK_NAME = "measured_mask_final_segmentation_hwatershed_bg500_90%.tif"
EXPECTED_SHAPE = (92, 31, 686, 608)


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT / path


def check_ome_channels(ome_xml, channel_indices):
    if not ome_xml:
        raise ValueError("OME metadata is missing")
    root = ET.fromstring(ome_xml)
    pixels = next((e for e in root.iter() if e.tag.endswith("Pixels")), None)
    if pixels is None:
        raise ValueError("OME Pixels metadata not found")
    names = [e.attrib.get("Name") for e in pixels if e.tag.endswith("Channel")]
    if len(names) != EXPECTED_SHAPE[1]:
        raise ValueError(f"Expected 31 named channels, got {len(names)}")
    expected = {"HER2_3B5": "Eu151", "HER2_29D8": "Gd156"}
    for marker, metal in expected.items():
        index = int(channel_indices[marker])
        if not (0 <= index < len(names) and names[index] == metal):
            raise ValueError(f"Channel mismatch for {marker}: index={index}, observed={names[index] if 0 <= index < len(names) else 'OUT_OF_RANGE'}")
    size_keys = (
        "PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ",
        "PhysicalSizeXUnit", "PhysicalSizeYUnit", "PhysicalSizeZUnit",
    )
    return {k: pixels.attrib.get(k) for k in size_keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=RUN_REL, help="Workflow 12E output directory")
    parser.add_argument("--base", default=BASE_REL, help="Raw SecondHer2 image directory")
    parser.add_argument("--qc", default=QC_REL, help="Prior identity QC directory")
    parser.add_argument("--out", default=OUT_REL, help="New output directory; must not exist")
    args = parser.parse_args()

    run = resolve_path(args.run)
    base = resolve_path(args.base)
    qc = resolve_path(args.qc)
    out = resolve_path(args.out)
    ome = base / OME_NAME
    mask = base / MASK_NAME
    summary_path = run / "summary.json"
    metrics_path = run / "candidate_boundary_metrics.csv"
    identity_path = qc / "secondher2_raw_instance_identity_qc.csv"

    for p in (ome, mask, summary_path, metrics_path, identity_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required input missing: {p}")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset") != "SecondHer2BreastCancerModel":
        raise ValueError("Unexpected dataset in Workflow 12E summary")
    x0, x1, y0, y1 = map(int, summary["roi_xy"])
    zs = [int(z) for z in summary["z_indices"]]
    if zs != list(range(EXPECTED_SHAPE[0])):
        raise ValueError("This full-3D script requires all consecutive Z indices 0..91")
    if not (0 <= x0 < x1 <= 608 and 0 <= y0 < y1 <= 686):
        raise ValueError("ROI outside expected image extent")
    roi_shape = (y1 - y0, x1 - x0)
    channels = {m: int(summary["channel_index"][m]) for m in ("HER2_3B5", "HER2_29D8")}

    mdf = pd.read_csv(metrics_path)
    baseline = mdf.loc[mdf["method"] == "nuclear_voronoi", "source_pseudo_id"]
    if baseline.isna().any():
        raise ValueError("Missing source IDs in Workflow 12E metrics")
    kept_ids = set(baseline.astype(np.int64).tolist())
    identity = pd.read_csv(identity_path)
    required = {"source_pseudo_id", "components_6", "components_26"}
    if not required.issubset(identity.columns):
        raise ValueError(f"Identity QC missing columns: {sorted(required - set(identity.columns))}")
    if identity.source_pseudo_id.duplicated().any():
        raise ValueError("Duplicate source IDs in identity QC")

    accum = {}
    section_rows = []
    with tifffile.TiffFile(ome) as tf, tifffile.TiffFile(mask) as mf:
        series = tf.series[0]
        if series.axes != "ZCYX" or tuple(series.shape) != EXPECTED_SHAPE:
            raise ValueError(f"Unexpected OME shape/axes: {series.shape}, {series.axes}")
        if len(series.pages) != 92 * 31 or len(mf.pages) != 92:
            raise ValueError("Unexpected OME or mask page count")
        physical = check_ome_channels(tf.ome_metadata, channels)
        print("PROJECT:", PROJECT)
        print("OME physical sizes (may be missing):", physical, flush=True)

        for z in zs:
            lab = mf.pages[z].asarray()[y0:y1, x0:x1]
            if lab.shape != roi_shape or not np.issubdtype(lab.dtype, np.integer):
                raise ValueError(f"Invalid mask geometry/type at z={z}: {lab.shape}, {lab.dtype}")
            if lab.min() < 0:
                raise ValueError(f"Negative source labels at z={z}")

            images = {}
            for marker, ci in channels.items():
                img = series.pages[z * 31 + ci].asarray()[y0:y1, x0:x1]
                if img.shape != roi_shape:
                    raise ValueError(f"Signal geometry mismatch at z={z} marker={marker}")
                images[marker] = img.astype(np.float64, copy=False)
                if not np.isfinite(images[marker]).all():
                    raise ValueError(f"Nonfinite intensity at z={z} marker={marker}")

            foreground = lab > 0
            if not foreground.any():
                section_rows.append({"z": z, "n_present_ids": 0, "n_labeled_voxels": 0,
                                     "HER2_3B5_mean": np.nan, "HER2_29D8_mean": np.nan})
                continue

            ids, inverse, counts = np.unique(lab[foreground], return_inverse=True, return_counts=True)
            sums = {}
            for marker, img in images.items():
                values = img[foreground]
                sums[marker] = (
                    np.bincount(inverse, weights=values, minlength=len(ids)),
                    np.bincount(inverse, weights=values ** 2, minlength=len(ids)),
                    np.bincount(inverse, weights=(values > 0).astype(np.float64), minlength=len(ids)),
                )

            border = np.concatenate((lab[0], lab[-1], lab[:, 0], lab[:, -1]))
            edge_ids = set(map(int, np.unique(border[border > 0])))
            for i, source_id in enumerate(ids):
                sid = int(source_id)
                row = accum.setdefault(sid, {
                    "source_pseudo_id": sid, "n_voxels": 0, "n_sections": 0,
                    "z_min": z, "z_max": z, "touches_xy_roi_edge": False,
                    "touches_z_stack_edge": False,
                })
                row["n_voxels"] += int(counts[i])
                row["n_sections"] += 1
                row["z_min"] = min(row["z_min"], z)
                row["z_max"] = max(row["z_max"], z)
                row["touches_xy_roi_edge"] |= sid in edge_ids
                row["touches_z_stack_edge"] |= z == 0 or z == 91
                for marker in channels:
                    for suffix, arr in zip(("sum", "sum_sq", "nonzero_count"), sums[marker]):
                        key = f"{marker}_{suffix}"
                        row[key] = row.get(key, 0.0) + float(arr[i])

            section_rows.append({
                "z": z, "n_present_ids": len(ids), "n_labeled_voxels": int(foreground.sum()),
                "HER2_3B5_mean": float(images["HER2_3B5"][foreground].mean()),
                "HER2_29D8_mean": float(images["HER2_29D8"][foreground].mean()),
            })
            if z % 10 == 0 or z == 91:
                print(f"Processed z={z}/91", flush=True)

    if not accum:
        raise ValueError("No positive source IDs found")
    result = pd.DataFrame(accum.values()).sort_values("source_pseudo_id").reset_index(drop=True)
    n = result.n_voxels.to_numpy(dtype=np.float64)
    for marker in channels:
        mean = result[f"{marker}_sum"].to_numpy() / n
        variance = result[f"{marker}_sum_sq"].to_numpy() / n - mean ** 2
        result[f"{marker}_mean"] = mean
        result[f"{marker}_std"] = np.sqrt(np.maximum(variance, 0))
        result[f"{marker}_nonzero_fraction"] = result[f"{marker}_nonzero_count"] / n

    result["internal_z_gaps"] = result.z_max - result.z_min + 1 - result.n_sections
    result["retained_by_workflow12e"] = result.source_pseudo_id.isin(kept_ids)
    result = result.merge(identity[["source_pseudo_id", "components_6", "components_26"]],
                          on="source_pseudo_id", how="left", validate="one_to_one")
    if result.loc[result.retained_by_workflow12e, "components_6"].isna().any():
        raise ValueError("Retained IDs missing prior 3D connectivity audit")

    def report(label, frame):
        print(f"\n{label}: {len(frame)} objects")
        if frame.empty:
            return
        print("Median voxels:", float(frame.n_voxels.median()))
        print("Median Z sections:", float(frame.n_sections.median()))
        if len(frame) >= 3:
            rho = frame[["HER2_3B5_mean", "HER2_29D8_mean"]].corr(method="spearman").iloc[0, 1]
            print("HER2 clone object-mean Spearman rho (exploratory):", float(rho))

    print("\n=== 3D HER2 PSEUDO-INSTANCE QUANTIFICATION ===")
    report("All raw ROI IDs", result)
    report("Workflow 12E retained", result[result.retained_by_workflow12e])
    eligible = result.loc[(result.retained_by_workflow12e)
                          & (result.components_6 == 1)
                          & (result.n_sections >= 2)
                          & (~result.touches_xy_roi_edge)]
    report("Exploratory geometry-filtered subset", eligible)
    print("Raw IDs not retained:", int((~result.retained_by_workflow12e).sum()))
    print("XY-edge touching:", int(result.touches_xy_roi_edge.sum()))
    print("Z-stack-edge touching:", int(result.touches_z_stack_edge.sum()))

    # Write output only after all computations and validations succeed.
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    with tempfile.TemporaryDirectory(prefix=".secondher2_3d_quant_", dir=out.parent) as td:
        stage = Path(td) / "results"
        stage.mkdir()
        result.to_csv(stage / "3d_her2_object_quantification.csv", index=False)
        pd.DataFrame(section_rows).to_csv(stage / "per_z_her2_intensity_qc.csv", index=False)
        provenance = {
            "analysis": "Exploratory raw 3D pseudo-instance HER2 mean intensity; NOT validated cells",
            "source_ome": str(ome), "source_mask": str(mask),
            "source_workflow12e": str(run), "roi_xy": [x0, x1, y0, y1],
            "n_z_sections": len(zs), "channel_index": channels,
            "ome_physical_sizes": physical,
            "n_raw_ids": len(result), "n_retained_ids": int(result.retained_by_workflow12e.sum()),
            "limitations": [
                "Author-provided Watershed labels are unvalidated pseudo-instances",
                "Mean intensity over a source region is not membrane-specific HER2 expression",
                "Nonzero fraction is not HER2 positivity; no threshold is defined",
                "Same-volume object correlations are exploratory, not independent sample replication",
            ],
        }
        (stage / "analysis_metadata.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(out)
    print("\nSUCCESS: output saved to", out)


if __name__ == "__main__":
    main()
