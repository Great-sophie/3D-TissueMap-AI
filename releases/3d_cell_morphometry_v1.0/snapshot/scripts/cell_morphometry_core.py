"""3D-TissueMap AI / Workflow 07 — pseudo-instance voxel morphometry.

Scientific contract:
- Input is a 3D integer instance-ID mask in explicitly ASSUMED Z,Y,X order.
- Labels are author-generated watershed PSEUDO-REFERENCES, not expert truth.
- No physical z spacing, physical surface, sphericity, protein-to-ID join, or
  specimen-level biological inference is calculated.
- All measurements use the exact source mask, without label repair or relabeling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import shutil
import tempfile
import sys
import json

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

VERSION = "1.0.0"
REQUIRED_COLUMNS = ("instance_id", "voxel_count", "centroid_z_index", "centroid_y_px", "centroid_x_px")


def validate_mask(mask: np.ndarray, max_label_id: int = 2_000_000) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 3 or any(int(d) < 1 for d in mask.shape):
        raise ValueError(f"Expected nonempty 3D (Z,Y,X) label mask; got shape {mask.shape}")
    if not np.issubdtype(mask.dtype, np.integer) or np.issubdtype(mask.dtype, np.bool_):
        raise ValueError(f"Expected INTEGER instance IDs, not {mask.dtype}")
    if np.issubdtype(mask.dtype, np.signedinteger) and np.any(mask < 0):
        raise ValueError("Negative instance labels are not allowed; 0 must be background")
    maximum = int(mask.max())
    if maximum == 0:
        raise ValueError("Mask has no nonzero instance IDs")
    if maximum > max_label_id:
        raise ValueError(f"Largest ID {maximum} exceeds safety guard {max_label_id}. "
                         "Use an explicit label remapping table instead of allocating a huge bincount.")
    return mask


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _accumulate(counts: np.ndarray, side: np.ndarray, n: int) -> None:
    """Increment per-label counts of faces owned by a nonzero label."""
    if side.size:
        counts += np.bincount(side.astype(np.int64, copy=False), minlength=n)


def count_exposed_faces(mask: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Count boundary voxel faces, including outside-volume faces, by Z/Y/X.

    An interface between two distinct nonzero IDs is counted once for EACH
    instance, but not twice within the same instance. Not physical contact area.
    """
    all_faces = np.zeros((3, n), dtype=np.int64)
    bg_faces = np.zeros((3, n), dtype=np.int64)
    other_faces = np.zeros((3, n), dtype=np.int64)
    # Compare along one axis, one Z chunk at a time; bounded peak memory.
    for axis in range(3):
        for z in range(mask.shape[0]):
            if axis == 0:
                if z == mask.shape[0] - 1:
                    continue
                a, b = mask[z], mask[z + 1]
            elif axis == 1:
                a, b = mask[z, :-1, :], mask[z, 1:, :]
            else:
                a, b = mask[z, :, :-1], mask[z, :, 1:]
            different = a != b
            if not different.any():
                continue
            for owner, neighbor in ((a, b), (b, a)):
                own_boundary = different & (owner != 0)
                if not own_boundary.any():
                    continue
                _accumulate(all_faces[axis], owner[own_boundary], n)
                _accumulate(bg_faces[axis], owner[own_boundary & (neighbor == 0)], n)
                _accumulate(other_faces[axis], owner[own_boundary & (neighbor != 0)], n)
        # Add outside-of-volume faces (all considered to border background).
        outside = ((mask[0], mask[-1]) if axis == 0 else
                   (mask[:, 0, :], mask[:, -1, :]) if axis == 1 else
                   (mask[:, :, 0], mask[:, :, -1]))
        for side in outside:
            positive = side[side > 0]
            _accumulate(all_faces[axis], positive, n)
            _accumulate(bg_faces[axis], positive, n)
    assert np.array_equal(all_faces, bg_faces + other_faces)
    return {"all": all_faces, "bg": bg_faces, "other": other_faces}


def summarize_mask(mask: np.ndarray, min_qc_voxels: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return per-instance morphology, per-Z slice areas, QC, and summary.

    This function is deterministic, modifies no voxels, and never joins protein data.
    """
    mask = validate_mask(mask)
    if min_qc_voxels < 1:
        raise ValueError("min_qc_voxels must be >= 1")
    Z, Y, X = map(int, mask.shape)
    maximum = int(mask.max())
    n = maximum + 1
    total_counts = np.bincount(mask.reshape(-1).astype(np.int64, copy=False), minlength=n)
    ids = np.flatnonzero(total_counts[1:]) + 1
    slices = ndi.find_objects(mask)  # bounding boxes; ID k corresponds to index k-1

    sumz = np.zeros(n, np.float64)
    sumy = np.zeros(n, np.float64)
    sumx = np.zeros(n, np.float64)
    n_planes = np.zeros(n, np.int32)
    area_min = np.full(n, np.iinfo(np.int64).max, np.int64)
    area_max = np.zeros(n, np.int64)
    area_squares = np.zeros(n, np.float64)
    area_peak_z = np.full(n, -1, np.int32)
    centroid_xy_step_max = np.zeros(n, np.float64)
    last_xy_y = np.full(n, np.nan, dtype=np.float64)
    last_xy_x = np.full(n, np.nan, dtype=np.float64)
    last_xy_z = np.full(n, -10, dtype=np.int32)
    slice_rows: list[dict] = []

    for z in range(Z):
        ys, xs = np.nonzero(mask[z])
        if not len(ys):
            continue
        labels = mask[z, ys, xs].astype(np.int64, copy=False)
        areas = np.bincount(labels, minlength=n)
        sum_y = np.bincount(labels, weights=ys.astype(np.float64), minlength=n)
        sum_x = np.bincount(labels, weights=xs.astype(np.float64), minlength=n)
        present = np.flatnonzero(areas[1:]) + 1
        n_planes[present] += 1
        area_min[present] = np.minimum(area_min[present], areas[present])
        area_squares[present] += areas[present].astype(np.float64)**2
        better_peak = areas[present] > area_max[present]
        area_peak_z[present[better_peak]] = z
        area_max[present] = np.maximum(area_max[present], areas[present])
        sumz[present] += areas[present].astype(np.float64) * z
        sumy[present] += sum_y[present]
        sumx[present] += sum_x[present]
        cy, cx = sum_y[present] / areas[present], sum_x[present] / areas[present]
        adjacent = last_xy_z[present] == z - 1
        if adjacent.any():
            jj = present[adjacent]
            d = np.hypot(cy[adjacent] - last_xy_y[jj], cx[adjacent] - last_xy_x[jj])
            centroid_xy_step_max[jj] = np.maximum(centroid_xy_step_max[jj], d)
        last_xy_y[present], last_xy_x[present], last_xy_z[present] = cy, cx, z
        for k, area, yc, xc in zip(present, areas[present], cy, cx):
            slice_rows.append({"instance_id": int(k), "z_index": int(z),
                               "area_px2": int(area), "centroid_y_px": float(yc),
                               "centroid_x_px": float(xc)})

    faces = count_exposed_faces(mask, n)
    structure6 = ndi.generate_binary_structure(3, 1)
    structure26 = ndi.generate_binary_structure(3, 3)
    morph_rows: list[dict] = []
    qc_rows: list[dict] = []
    for k in ids:
        k = int(k)
        bbox = slices[k - 1]
        if bbox is None:
            raise RuntimeError(f"Missing bbox for present instance {k}")
        (z0, z1), (y0, y1), (x0, x1) = [(s.start, s.stop) for s in bbox]
        voxels = int(total_counts[k])
        box_voxels = int((z1 - z0) * (y1 - y0) * (x1 - x0))
        box = mask[bbox] == k
        cc6_labels, cc6_count = ndi.label(box, structure=structure6)
        _, cc26_count = ndi.label(box, structure=structure26)
        largest6 = int(np.bincount(cc6_labels.ravel())[1:].max()) if cc6_count else 0
        zspan = int(z1 - z0)
        slices_seen = int(n_planes[k])
        mean_area = float(voxels / slices_seen)
        sq_mean = area_squares[k] / slices_seen
        std_area = float(np.sqrt(max(0.0, sq_mean - mean_area**2)))
        fz, fy, fx = (int(faces["all"][ax, k]) for ax in range(3))
        bgz, bgy, bgx = (int(faces["bg"][ax, k]) for ax in range(3))
        oz, oy, ox = (int(faces["other"][ax, k]) for ax in range(3))
        row = {
            "instance_id": k, "voxel_count": voxels,
            "centroid_z_index": float(sumz[k] / voxels),
            "centroid_y_px": float(sumy[k] / voxels),
            "centroid_x_px": float(sumx[k] / voxels),
            "z_min_index": int(z0), "z_max_index": int(z1 - 1),
            "y_min_px": int(y0), "y_max_px": int(y1 - 1),
            "x_min_px": int(x0), "x_max_px": int(x1 - 1),
            "z_span_planes": zspan, "n_slices_present": slices_seen,
            "z_gap_planes": int(zspan - slices_seen),
            "bbox_height_px": int(y1 - y0), "bbox_width_px": int(x1 - x0),
            "bbox_voxels": box_voxels, "bbox_occupancy_fraction": float(voxels / box_voxels),
            "mean_slice_area_px2": mean_area,
            "min_slice_area_px2": int(area_min[k]),
            "max_slice_area_px2": int(area_max[k]),
            "peak_area_z_index": int(area_peak_z[k]),
            "slice_area_cv": std_area / mean_area if mean_area else float("nan"),
            "max_adjacent_plane_centroid_shift_xy_px": float(centroid_xy_step_max[k]),
            "bbox_xy_aspect_ratio": float(max(y1-y0, x1-x0) / min(y1-y0, x1-x0)),
            "exposed_z_faces_voxel": fz, "exposed_y_faces_voxel": fy,
            "exposed_x_faces_voxel": fx, "exposed_all_faces_voxel": fz + fy + fx,
            "background_faces_voxel": bgz + bgy + bgx,
            "other_instance_interface_faces_voxel": oz + oy + ox,
            "surface_faces_per_voxel_to_two_thirds": float((fz+fy+fx)/(voxels**(2.0/3.0))),
        }
        morph_rows.append(row)
        at_z_border = bool(z0 == 0 or z1 == Z)
        at_xy_border = bool(y0 == 0 or y1 == Y or x0 == 0 or x1 == X)
        issues = []
        if cc6_count > 1:
            issues.append("MULTIPLE_CC6")
        if int(zspan - slices_seen) > 0:
            issues.append("MISSING_INTERMEDIATE_Z")
        if at_z_border:
            issues.append("TOUCHES_Z_VOLUME_BOUNDARY")
        if at_xy_border:
            issues.append("TOUCHES_XY_VOLUME_BOUNDARY")
        if voxels < min_qc_voxels:
            issues.append("SMALL_PSEUDO_INSTANCE_HEURISTIC")
        qc_rows.append({"instance_id": k, "components_6connected": int(cc6_count),
                        "components_26connected": int(cc26_count),
                        "largest_cc6_voxel_fraction": float(largest6/voxels),
                        "touches_z_stack_boundary": at_z_border,
                        "touches_xy_image_boundary": at_xy_border,
                        "qc_flag_review": bool(issues),
                        "qc_reasons": "|".join(issues),
                        "note": "Heuristic QC on author pseudo-instances; not biological validation"})

    morph = pd.DataFrame(morph_rows).sort_values("instance_id", ignore_index=True)
    area_profile = pd.DataFrame(slice_rows, columns=["instance_id","z_index","area_px2","centroid_y_px","centroid_x_px"])
    if not area_profile.empty:
        area_profile = area_profile.sort_values(["instance_id", "z_index"], ignore_index=True)
    qc = pd.DataFrame(qc_rows).sort_values("instance_id", ignore_index=True)
    summary = {
        "version": VERSION,
        "provenance": "Author automated watershed 3D instance pseudo-labels (not manual cell ground truth)",
        "axis_assumption": "ZYX assumed explicitly, not physically validated",
        "shape_assumed_zyx": [Z,Y,X],
        "n_instance_ids": int(len(ids)),
        "foreground_voxels": int(total_counts[ids].sum()),
        "foreground_fraction": float(int(total_counts[ids].sum()) / mask.size),
        "n_multiplane_ids": int((morph.n_slices_present > 1).sum()),
        "n_z_gapped_ids": int((morph.z_gap_planes > 0).sum()),
        "n_multicomponent_6": int((qc.components_6connected > 1).sum()),
        "n_multicomponent_26": int((qc.components_26connected > 1).sum()),
        "n_touching_z_boundary": int(qc.touches_z_stack_boundary.sum()),
        "n_touching_xy_boundary": int(qc.touches_xy_image_boundary.sum()),
        "n_qc_review_flagged": int(qc.qc_flag_review.sum()),
        "min_qc_voxels_parameter": int(min_qc_voxels),
        "voxel_count_quantiles_0_25_50_75_90_99_100": [float(q) for q in
            np.quantile(morph.voxel_count, [0, .25, .5, .75, .9, .99, 1])],
        "interpretation_limits": [
            "No verified XY pixel size or physical Z spacing: no physical size/area/volume reported.",
            "No physical 3D sphericity or anisotropy-sensitive geometric claim.",
            "Surface-face counts are voxel-grid interfaces, NOT calibrated surface areas or cell-cell contacts.",
            "6/26-connectivity and ID continuity are internal properties of automated watershed pseudo-labels.",
            "Reference ID identity, image registration and protein-expression mapping are not independently verified.",
            "Single specimen: no independent specimen-level generalization; no biological conclusions.",
            "Small-instance QC thresholds are heuristics, not tissue-specific clinical cutoffs.",
        ]
    }
    assert len(morph) == len(qc) == len(ids)
    assert int(area_profile.area_px2.sum()) == summary["foreground_voxels"]
    assert np.all(morph.voxel_count.to_numpy() == area_profile.groupby("instance_id").area_px2.sum().reindex(ids).to_numpy())
    assert np.all(morph.exposed_all_faces_voxel.to_numpy() == (morph.background_faces_voxel+morph.other_instance_interface_faces_voxel).to_numpy())
    return morph, area_profile, qc, summary


def draw_figures(morph: pd.DataFrame, profiles: pd.DataFrame, qc: pd.DataFrame,
                 mask: np.ndarray, output: Path) -> list[str]:
    """Make scientific QC plots; selected 3D examples use voxel indices, NOT µm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    files = []
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.hist(np.log10(morph.voxel_count.to_numpy()), bins=min(65,max(8,int(np.sqrt(len(morph))))))
    ax.set_xlabel("log10(instance voxel count)")
    ax.set_ylabel("Author pseudo-instance count")
    ax.set_title("Pseudo-instance voxel counts — not physical volume")
    fig.tight_layout()
    fn = "instance_voxel_count_distribution.png"
    fig.savefig(output/fn, dpi=145)
    plt.close(fig)
    files.append(fn)

    fig, ax = plt.subplots(figsize=(7.6,4.6))
    vals, num = np.unique(morph.z_span_planes.to_numpy(), return_counts=True)
    ax.bar(vals, num)
    ax.set_xlabel("Z span (number of index planes, NOT distance)")
    ax.set_ylabel("Author pseudo-instance count")
    ax.set_title("Z occupancy and continuity")
    fig.tight_layout()
    fn = "z_span_distribution.png"
    fig.savefig(output/fn, dpi=145)
    plt.close(fig)
    files.append(fn)

    fig, ax = plt.subplots(figsize=(7.6,4.6))
    qccounts = qc.qc_flag_review.value_counts()
    ax.bar(["No heuristic flags", "Review suggested"],
           [int(qccounts.get(False,0)),int(qccounts.get(True,0))])
    ax.set_ylabel("Author pseudo-instance count")
    ax.set_title("Morphological QC — flags are not biological errors")
    fig.tight_layout()
    fn = "qc_flag_summary.png"
    fig.savefig(output/fn, dpi=145)
    plt.close(fig)
    files.append(fn)

    eligible = morph[(morph.n_slices_present > 1)&(morph.voxel_count.between(8,2500))]
    if not eligible.empty:
        select = eligible.sort_values("voxel_count").iloc[len(eligible)//2]
        chosen_id = int(select.instance_id)
        sub = profiles[profiles.instance_id == chosen_id]
        fig, ax = plt.subplots(figsize=(7.6,4.6))
        ax.plot(sub.z_index, sub.area_px2, marker="o")
        ax.set_xlabel("Z plane index (not physical depth)")
        ax.set_ylabel("Area (XY pixels)")
        ax.set_title(f"Example slice area profile: pseudo-ID {chosen_id}")
        fig.tight_layout()
        fn = "example_z_area_profile.png"
        fig.savefig(output/fn, dpi=145)
        plt.close(fig)
        files.append(fn)

        # One sparse 3D representation with explicit pixel/plane coordinate axes.
        zs, ys, xs = np.where(mask == chosen_id)
        if len(xs) > 1200:
            select_index = np.linspace(0,len(xs)-1,1200).astype(int)
            xs, ys, zs = xs[select_index], ys[select_index], zs[select_index]
        fig = plt.figure(figsize=(7.2,5.6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs,ys,zs, s=7, alpha=.48, depthshade=False)
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        ax.set_zlabel("Z plane index")
        ax.set_title(f"Voxel-coordinate preview: pseudo-ID {chosen_id}\nNOT physical 3D aspect ratio")
        fig.tight_layout()
        fn = "example_3d_voxel_cloud_uncalibrated.png"
        fig.savefig(output/fn,dpi=145)
        plt.close(fig)
        files.append(fn)
    return files


def save_tables(mask: np.ndarray, source_path: Path, out: Path,
                min_qc_voxels: int = 12, figures: bool = True,
                tiff_series_axes: str | None = None) -> dict:
    """Write a new run atomically. Never overwrite, alter, or repair a source mask."""
    if out.exists():
        raise FileExistsError(f"Output already exists (no implicit overwrite): {out}")
    morph, profile, qc, summary = summarize_mask(mask, min_qc_voxels=min_qc_voxels)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".morphometry_v1_staging_", dir=out.parent))
    try:
        morph.to_csv(stage/"cell_morphometry.csv", index=False, float_format="%.8g")
        # shape-specific columns are separate but same stable instance ID.
        shape_columns = ["instance_id", "z_span_planes", "n_slices_present", "z_gap_planes",
            "bbox_occupancy_fraction", "bbox_xy_aspect_ratio", "mean_slice_area_px2",
            "min_slice_area_px2", "max_slice_area_px2", "peak_area_z_index",
            "slice_area_cv", "max_adjacent_plane_centroid_shift_xy_px",
            "exposed_z_faces_voxel", "exposed_y_faces_voxel", "exposed_x_faces_voxel",
            "exposed_all_faces_voxel", "background_faces_voxel",
            "other_instance_interface_faces_voxel", "surface_faces_per_voxel_to_two_thirds"]
        morph[shape_columns].to_csv(stage/"cell_shape_features.csv", index=False, float_format="%.8g")
        profile.to_csv(stage/"cell_slice_profiles.csv", index=False, float_format="%.8g")
        qc.to_csv(stage/"cell_morphology_qc.csv", index=False, float_format="%.8g")
        images = draw_figures(morph, profile, qc, mask, stage/"figures") if figures else []
        import numpy as _np
        import scipy as _scipy
        import pandas as _pd
        import tifffile as _tifffile
        summary.update({"source_mask_path": str(source_path.resolve()),
                        "source_mask_sha256": file_sha256(source_path),
                        "source_mask_dtype": str(mask.dtype),
                        "input_tiff_series_axes_reported": tiff_series_axes,
                        "software_versions": {"python": sys.version.split()[0], "numpy": _np.__version__,
                                              "scipy": _scipy.__version__, "pandas": _pd.__version__,
                                              "tifffile": _tifffile.__version__},
                        "core_code_sha256": file_sha256(Path(__file__)),
                        "requested_outputs": ["cell_morphometry.csv","cell_shape_features.csv",
                                              "cell_slice_profiles.csv","cell_morphology_qc.csv"],
                        "figure_files": images})
        (stage/"morphometry_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        # Checksums cover all generated files except the checksum index itself.
        manifest = {str(p.relative_to(stage)): file_sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        (stage/"output_file_sha256.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        # Staged directory is on same filesystem as destination: atomic rename.
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite a concurrent output: {out}")
        stage.rename(out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return summary
