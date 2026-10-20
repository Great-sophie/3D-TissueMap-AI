#!/usr/bin/env python3
"""Workflow 12E: frozen-parameter 2D candidate segmentation transfer to SecondHer2.

Reuses the *unchanged* Workflow 11H segment_slice(). Measured pseudo-nuclear
instances provide hard seeds; panCK epithelial domains and Yb173 cadherin
provide input cues. HER2_3B5 and HER2_29D8 are held-out readouts only.
NO 3D cell identity or ground-truth segmentation accuracy is claimed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Import original algorithms from the same scripts directory; never rewrite them.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from run_workflow11h_nucleus_constrained_segmentation import segment_slice
from run_workflow11g_cadherin_segmentation import region_metrics

VERSION = "0.1.0-fixed-param-cross-volume-pilot"
DEFAULT_BASE = Path("data/inspect/SecondHer2BreastCancerModel/SecondHer2BreastCancerModel")
DEFAULT_OUTPUT = Path("runs/workflow12_secondher2_fixed_boundary_transfer_pilot_v0_1")
METALS = {
    "Ir193": "Ir193",
    "panCK": "Lu175",
    "Cadherin": "Yb173",  # IMPORTANT: SecondHer2 != LVILymph (Yb174)
    "HER2_3B5": "Eu151",
    "HER2_29D8": "Gd156",
}
PARAMS = {
    "quantile": 0.80,
    "sigma": 1.0,
    "min_domain_px": 40,
    "min_seed_overlap_px": 4,
    "cadherin_weight": 0.75,
}
EVAL = {"min_interface_px": 3, "min_core_px": 5}
EXPECTED_SHAPE = (92, 31, 686, 608)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def parse_zs(z_str, n=92):
    zs = list(range(n)) if z_str == "all" else [int(x.strip()) for x in z_str.split(",") if x.strip()]
    if not zs or len(zs) != len(set(zs)) or sorted(zs) != zs or any(z < 0 or z >= n for z in zs):
        raise ValueError("--zs must be sorted, unique and in [0,91], or 'all'")
    return zs


def preflight(a):
    base = Path(a.base)
    ome = base / "aligned_SecondHer2BreastCancer_model_3D_image_stack.ome.tiff"
    mask = base / "measured_mask_final_segmentation_hwatershed_bg500_90%.tif"
    script_11h = SCRIPTS_DIR / "run_workflow11h_nucleus_constrained_segmentation.py"
    script_11g = SCRIPTS_DIR / "run_workflow11g_cadherin_segmentation.py"
    script_11e = SCRIPTS_DIR / "run_workflow11e_membrane_feasibility.py"
    script_11a = SCRIPTS_DIR / "run_spatial_signal_attribution.py"
    script_11core = SCRIPTS_DIR / "spatial_signal_attribution_core.py"
    dependencies = [script_11h, script_11g, script_11e, script_11a, script_11core]
    for p in [ome, mask, *dependencies]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing prerequisite: {p}")
    zs = parse_zs(a.zs, n=EXPECTED_SHAPE[0])
    x0, x1, y0, y1 = a.roi
    if not (0 <= x0 < x1 <= EXPECTED_SHAPE[3] and 0 <= y0 < y1 <= EXPECTED_SHAPE[2]):
        raise ValueError("--roi X0 X1 Y0 Y1 must fit measured XY 608 x 686")
    with tifffile.TiffFile(ome) as tf, tifffile.TiffFile(mask) as mf:
        s = tf.series[0]
        if s.axes != "ZCYX" or s.shape != EXPECTED_SHAPE or len(s.pages) != EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1]:
            raise ValueError(f"OME axis/shape/pages unexpected: {s.axes}, {s.shape}, {len(s.pages)}")
        if len(mf.pages) != EXPECTED_SHAPE[0] or mf.pages[0].shape != EXPECTED_SHAPE[2:]:
            raise ValueError("Measured mask geometry mismatch")
        if not tf.ome_metadata:
            raise ValueError("Missing OME channel metadata")
        root = ET.fromstring(tf.ome_metadata)
        pixels = next(e for e in root.iter() if e.tag.endswith("Pixels"))
        names = [c.attrib.get("Name") for c in pixels if c.tag.endswith("Channel")]
        if len(names) != EXPECTED_SHAPE[1] or len(set(names)) != EXPECTED_SHAPE[1]:
            raise ValueError("Invalid OME channel metadata")
        channel_index = {role: names.index(metal) for role, metal in METALS.items()}
        return dict(ome=ome, mask=mask, channels=channel_index, zs=zs, roi=list(a.roi),
                    dependencies=dependencies, ome_names=names)


def make_geometry(metrics):
    d = pd.DataFrame(metrics)
    if len(d) == 0:
        raise RuntimeError("No retained seeded candidates; inspect panCK domain or nuclear mask")
    out = {}
    for method, g in d.groupby("method"):
        out[method] = {
            "n_segments": len(g),
            "testable": int(g.testable_interface.sum()),
            "fraction_testable": float(g.testable_interface.mean()),
            "median_area_px": float(g.segment_area_px.median()),
            "median_core_px": float(g.n_core_px.median()),
            "n_area_le_5": int((g.segment_area_px <= 5).sum()),
            "n_less_than_nuclear_fragment": int((g.segment_area_px < g.nucleus_fragment_px).sum()),
        }
    return out


def run(a):
    meta = preflight(a)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    x0, x1, y0, y1 = a.roi
    key = ["z", "seed_id", "method"]
    metrics = []
    seed_rows = []
    z_qc = []
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".wf12e_stage_", dir=out.parent))
    try:
        (stage / "candidate_labels").mkdir()
        with tifffile.TiffFile(meta["ome"]) as tf, tifffile.TiffFile(meta["mask"]) as mf:
            pages = tf.series[0].pages
            for z in meta["zs"]:
                nuclei = mf.pages[z].asarray()[y0:y1, x0:x1]
                images = {}
                for marker, ci in meta["channels"].items():
                    images[marker] = pages[z * EXPECTED_SHAPE[1] + ci].asarray()[y0:y1, x0:x1]
                domain, markers, seeds, baseline, assisted, threshold, domains = segment_slice(
                    images["panCK"], nuclei, images["Cadherin"], **PARAMS
                )
                if not np.array_equal(baseline > 0, assisted > 0):
                    raise RuntimeError(f"Z={z}: method domain coverage changed")
                for method, labels in [("nuclear_voronoi", baseline), ("cadherin_assisted", assisted)]:
                    if not np.array_equal(labels[markers > 0], markers[markers > 0]):
                        raise RuntimeError(f"Z={z}: {method} has lost nuclear fragment pixels")
                    tifffile.imwrite(stage / "candidate_labels" / f"z{z:03d}_{method}.tif",
                                     labels.astype(np.uint32), photometric="minisblack")
                    # Reuse 11G's frozen interface/core definition for each held-out HER2 clone.
                    m_a = region_metrics(labels, images["HER2_3B5"], images["Cadherin"],
                                         seeds, method, **EVAL)
                    m_b = region_metrics(labels, images["HER2_29D8"], images["Cadherin"],
                                         seeds, method, **EVAL)
                    if len(m_a) != len(seeds) or len(m_b) != len(seeds):
                        raise RuntimeError("Inconsistent metric mapping")
                    for ra, rb in zip(m_a, m_b):
                        if ra["seed_id"] != rb["seed_id"]:
                            raise RuntimeError("Heldout readout ID mismatch")
                        ra["z"] = z
                        ra["HER2_3B5_interface_minus_core"] = ra.pop("HER2_interface_minus_core")
                        ra["HER2_3B5_interface_mean"] = ra.pop("HER2_interface_mean")
                        ra["HER2_3B5_core_mean"] = ra.pop("HER2_core_mean")
                        ra["HER2_29D8_interface_minus_core"] = rb["HER2_interface_minus_core"]
                        ra["HER2_29D8_interface_mean"] = rb["HER2_interface_mean"]
                        ra["HER2_29D8_core_mean"] = rb["HER2_core_mean"]
                        metrics.append(ra)
                for seed in seeds:
                    seed_rows.append({"z": z, **seed})
                z_qc.append({"z": z, "n_nuclear_fragments": len(seeds),
                             "n_domains": len(domains), "n_seeded_domains": sum(v["n_seed_fragments"] > 0 for v in domains),
                             "panCK_log_threshold": threshold,
                             "protected_nuclear_px": int((markers > 0).sum()),
                             "segmented_domain_px": int((baseline > 0).sum()),
                             "coverage_equal": bool(np.array_equal(baseline > 0, assisted > 0))})
                print(f"Z={z:02d}/91: {len(seeds)} protected fragments | assigned domain px={int((baseline>0).sum())}", flush=True)
        if not metrics:
            raise RuntimeError("No seed fragments in the chosen ROI/Z combination")
        d = pd.DataFrame(metrics)
        if d.duplicated(key).any() or len(d) != 2 * len(seed_rows):
            raise RuntimeError("Expected exactly 2 methods per nuclear seed")
        if any(int(v) != 0 for v in [
            ((d.segment_area_px < d.nucleus_fragment_px) & (d.method == method)).sum()
            for method in ("nuclear_voronoi", "cadherin_assisted")
        ]):
            raise RuntimeError("Nucleus size invariant violated")
        area = d.pivot(index=["z", "seed_id"], columns="method", values="segment_area_px")
        collapse = ((area["nuclear_voronoi"] >= 20) &
                    (area["cadherin_assisted"] <= 0.25 * area["nuclear_voronoi"]))
        usability = d.pivot(index=["z", "seed_id"], columns="method", values="testable_interface")
        both_eligible = usability["nuclear_voronoi"].astype(bool) & usability["cadherin_assisted"].astype(bool)
        summary = {
            "workflow": "12E-2D-slice-cross-volume-transfer", "version": VERSION,
            "dataset": "SecondHer2BreastCancerModel", "z_indices": meta["zs"], "roi_xy": meta["roi"],
            "channel_index": meta["channels"], "metals": METALS, "algorithm_params": PARAMS,
            "metric_params": EVAL, "n_section_nuclear_fragments": len(seed_rows),
            "n_selected_sections": len(meta["zs"]),
            "geometry_by_method": make_geometry(metrics),
            "n_severely_shrunken_vs_voronoi": int(collapse.sum()),
            "n_paired_testable": int(both_eligible.sum()),
            "n_paired_testable_fraction": float(both_eligible.mean()),
            "intensity_readout_status": "exploratory held-out signals; not segmentation accuracy",
            "limitations": ["2D slice-wise ROIs; NOT 3D whole-cell segmentation",
                            "measured nuclear-like pseudo-instances are not manually confirmed cells",
                            "Yb173 E/P Cadherin is algorithm input; not an independent boundary ground truth",
                            "HER2 clones were held out of mask construction, but spatial colocalization and sample selection remain confounders",
                            "fixed parameters have not been optimized for this dataset"],
        }
        pd.DataFrame(metrics).to_csv(stage / "candidate_boundary_metrics.csv", index=False)
        pd.DataFrame(seed_rows).to_csv(stage / "section_nuclear_seed_index.csv", index=False)
        pd.DataFrame(z_qc).to_csv(stage / "per_z_geometry_qc.csv", index=False)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        sources = [meta["ome"], meta["mask"], *meta["dependencies"], Path(__file__)]
        hashes = {str(p.resolve()): sha256(p) for p in sources}
        (stage / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        output_hashes = {str(f.relative_to(stage)): sha256(f) for f in stage.rglob("*") if f.is_file()}
        (stage / "output_sha256.json").write_text(json.dumps(output_hashes, indent=2) + "\n")
        stage.rename(out)
        print("\n===== WORKFLOW 12E SUMMARY =====")
        print(json.dumps({k:v for k,v in summary.items() if k in (
            "n_selected_sections", "n_section_nuclear_fragments", "geometry_by_method",
            "n_severely_shrunken_vs_voronoi", "n_paired_testable",
            "n_paired_testable_fraction")}, indent=2))
        print("OUTPUT:", out)
        print("Use --verify to validate hashes and invariants")
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify(a):
    out = Path(a.out)
    if not out.is_dir():
        raise FileNotFoundError(out)
    record = json.loads((out / "summary.json").read_text())
    meta = preflight(a)
    if record["z_indices"] != meta["zs"] or record["roi_xy"] != meta["roi"] or record["channel_index"] != meta["channels"]:
        raise RuntimeError("The requested ROI/Z/channel configuration differs from the frozen run")
    if record["algorithm_params"] != PARAMS or record["metric_params"] != EVAL:
        raise RuntimeError("The algorithm parameters changed")
    input_hashes = json.loads((out / "source_sha256.json").read_text())
    for path, expected in input_hashes.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Source integrity failure: {path}")
    output_hashes = json.loads((out / "output_sha256.json").read_text())
    actual = {str(f.relative_to(out)) for f in out.rglob("*") if f.is_file()}
    if actual != (set(output_hashes) | {"output_sha256.json"}):
        raise RuntimeError("Unexpected output file added/missing")
    for rel, expected in output_hashes.items():
        if sha256(out / rel) != expected:
            raise RuntimeError(f"Modified output: {rel}")
    d = pd.read_csv(out / "candidate_boundary_metrics.csv")
    if len(d) != 2 * record["n_section_nuclear_fragments"]:
        raise RuntimeError("Metric count mismatch")
    if (d.segment_area_px < d.nucleus_fragment_px).any():
        raise RuntimeError("Nuclear fragment area invariant violated")
    for row in pd.read_csv(out / "per_z_geometry_qc.csv").itertuples():
        if not row.coverage_equal:
            raise RuntimeError("Tissue domain assignment differs between methods")
    return {"status": "PASS", "verified_outputs": len(output_hashes),
            "verified_sources": len(input_hashes),
            "n_selected_sections": record["n_selected_sections"],
            "n_section_nuclear_fragments": record["n_section_nuclear_fragments"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--roi", type=int, nargs=4, default=(160, 416, 215, 471), metavar=("X0", "X1", "Y0", "Y1"))
    p.add_argument("--zs", default="0,23,45,68,91", help="Sorted comma-separated indices or 'all'")
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    a = p.parse_args()
    if a.check:
        meta = preflight(a)
        print(json.dumps({"status": "PREFLIGHT_PASS", "zs": meta["zs"], "roi_xy": meta["roi"],
                          "shape": EXPECTED_SHAPE, "channel_index": meta["channels"],
                          "fixed_algorithm_params": PARAMS,
                          "new_output": str(a.out)}, indent=2))
    elif a.apply:
        run(a)
    else:
        print(json.dumps(verify(a), indent=2))


if __name__ == "__main__":
    main()
