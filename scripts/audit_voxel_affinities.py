#!/usr/bin/env python3
"""Audit 3D adjacent-voxel same-instance supervision in reference masks.

Only uses chosen spatial split; no neural-network predictions or test-set metrics.
Reference labels are author-generated pseudo-labels, not manual ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

AXES = ("Z", "Y", "X")


def axis_affinity_stats(labels: np.ndarray, axis: int) -> dict:
    """Measure pairs along the positive direction of one voxel axis."""
    if labels.ndim != 3 or axis not in (0, 1, 2):
        raise ValueError("Expected a ZYX 3D label image and axis in 0,1,2")
    if min(labels.shape) < 2:
        raise ValueError("ZYX volume needs >=2 voxels per axis")
    sa = [slice(None)] * 3
    sb = [slice(None)] * 3
    sa[axis] = slice(None, -1)
    sb[axis] = slice(1, None)
    a, b = labels[tuple(sa)], labels[tuple(sb)]
    valid = (a > 0) & (b > 0)
    same = valid & (a == b)
    different = valid & (a != b)
    n_same, n_diff = int(np.count_nonzero(same)), int(np.count_nonzero(different))
    n_valid = n_same + n_diff
    return {
        "valid_cell_cell_faces": n_valid,
        "same_instance_faces": n_same,
        "different_instance_faces": n_diff,
        "different_fraction_among_cell_cell": float(n_diff / n_valid) if n_valid else None,
        "same_fraction_among_cell_cell": float(n_same / n_valid) if n_valid else None,
        "different_fraction_all_faces": float(n_diff / a.size),
    }


def adjacency_maps(labels: np.ndarray, z: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D views: different-ID transitions to next Z, Y, and X voxel."""
    if z < 0 or z >= labels.shape[0] - 1:
        raise ValueError("z out of range")
    zmap = ((labels[z] > 0) & (labels[z+1] > 0) & (labels[z] != labels[z+1]))
    ymap = np.zeros_like(zmap, dtype=bool)
    xmap = np.zeros_like(zmap, dtype=bool)
    a = labels[z, :-1, :]; b = labels[z, 1:, :]
    ymap[:-1, :] = ((a > 0) & (b > 0) & (a != b))
    a = labels[z, :, :-1]; b = labels[z, :, 1:]
    xmap[:, :-1] = ((a > 0) & (b > 0) & (a != b))
    return zmap, ymap, xmap


def audit(mask_file: Path, split_csv: Path, output: Path, split: str, guard: int):
    output.mkdir(parents=True, exist_ok=True)
    if split not in ("train", "val"):
        raise ValueError("Only train or val allowed, never test")
    with tifffile.TiffFile(mask_file) as f:
        shape = f.series[0].shape
        if len(shape) != 3:
            raise ValueError(f"Expected a 3D mask; got {shape}")
        labels = f.asarray()
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Instance labels must have integer dtype")
    tiles = pd.read_csv(split_csv)
    missing = {"split", "y", "x", "w"} - set(tiles.columns)
    if missing:
        raise ValueError(f"Missing split manifest columns: {sorted(missing)}")
    tiles = tiles[tiles["split"] == split]
    if tiles.empty:
        raise ValueError(f"No {split} tiles")

    rows = []
    preview = None
    for i, t in enumerate(tiles.itertuples(index=False), 1):
        y, x, w = int(t.y), int(t.x), int(t.w)
        if w <= 2*guard or x < 0 or y < 0 or x+w > labels.shape[2] or y+w > labels.shape[1]:
            raise ValueError(f"Invalid tile {i}: {y},{x},{w},guard={guard}")
        crop = labels[:, y+guard:y+w-guard, x+guard:x+w-guard]
        for axis, name in enumerate(AXES):
            rows.append({"tile": i, "tile_y": y, "tile_x": x, "axis": name,
                         "n_instances_in_crop": int(np.unique(crop[crop>0]).size),
                         "foreground_fraction": float(np.mean(crop>0)),
                         **axis_affinity_stats(crop, axis)})
        if preview is None:
            preview = crop.copy()
        print(f"{split} tile {i}/{len(tiles)} audited", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output / "per_tile_axis_affinities.csv", index=False)
    summaries = []
    for name in AXES:
        block = df[df.axis == name]
        same, different = int(block.same_instance_faces.sum()), int(block.different_instance_faces.sum())
        total = same + different
        summaries.append({"axis": name, "n_tiles": len(block), "same_instance_faces": same,
                          "different_instance_faces": different, "valid_cell_cell_faces": total,
                          "different_fraction_among_cell_cell": different / total if total else None})
    pd.DataFrame(summaries).to_csv(output / "axis_affinity_summary.csv", index=False)

    assert preview is not None
    z = min(preview.shape[0]-2, preview.shape[0]//2)
    zmap, ymap, xmap = adjacency_maps(preview,z)
    fig, axes = plt.subplots(1,4,figsize=(15,4))
    axes[0].imshow(preview[z],cmap="nipy_spectral",interpolation="nearest")
    for a, arr, title in zip(axes[1:], (zmap,ymap,xmap), ("different ID: Z+1", "different ID: Y+1", "different ID: X+1")):
        a.imshow(arr,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
        a.set_title(title)
    axes[0].set_title("Author pseudo-instance IDs")
    for a in axes:a.axis("off")
    fig.suptitle(f"{split.upper()} only | adjacent-voxel pseudo-affinity targets (not physical contact area)")
    fig.tight_layout();fig.savefig(output / "affinity_target_preview.png",dpi=140);plt.close(fig)

    result = {
       "scope": f"{split.upper()} XY tiles only, same 3D specimen; existing split manifest",
       "source": "Author-produced watershed instance pseudo-labels, NOT independently annotated GT",
       "shape_ZYX": list(map(int,labels.shape)),
       "n_tiles": int(len(tiles)),
       "axes": summaries,
       "targets": "For two foreground-adjacent voxels, affinity=1 if same nonzero instance ID, else 0",
       "masked_pairs": "Background-involved pairs omitted from the conditional cell-cell affinity class balance",
       "limitations": ["Voxel adjacency does not establish biological cell contact",
                       "Physical spacing and mask registration remain unverified",
                       "Adjacent edges highly correlated; counts are not independent samples",
                       "Reference labels may be flawed, especially across serial-section Z",
                       "No TEST tiles touched; this audit is not a model score"],
    }
    (output / "affinity_target_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
    print("Saved:",output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--split-run-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("runs/affinity_target_audit"))
    p.add_argument("--split",choices=("train","val"),default="train")
    p.add_argument("--guard",type=int,default=12)
    a=p.parse_args()
    fs=sorted(a.data_dir.glob("full_mask_final_segmentation*.tif"))
    if len(fs)!=1:raise FileNotFoundError(f"Expected one full_mask tif in {a.data_dir}; found {len(fs)}")
    audit(fs[0],a.split_run_dir/"tile_splits.csv",a.out,a.split,a.guard)

if __name__=="__main__":main()
