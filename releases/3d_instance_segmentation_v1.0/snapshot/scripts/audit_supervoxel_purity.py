#!/usr/bin/env python3
"""Reference-foreground supervoxel purity audit (diagnostic; NO model selection).

Reimplements the actual block_supervoxels() rule with 6-connected components
within each grid block. Reference labels serve ONLY to score the generated
supervoxels, never as inputs to the partition other than reference foreground.
This is an optimistic *perfect foreground* assessment, not predicted-foreground
or physical cell accuracy.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Reuse the exact code used by the previous Region Graph pilot.
from scipy import ndimage as ndi


def block_supervoxels(fg: np.ndarray, block_zyx=(2, 6, 6)) -> np.ndarray:
    """Exact algorithmic rule of Region Graph pilot, copied to stay self-contained."""
    fg = np.asarray(fg, bool)
    if fg.ndim != 3 or any(s < 1 for s in block_zyx):
        raise ValueError("fg must be 3D and block sizes positive")
    labels = np.zeros(fg.shape, dtype=np.int32)
    current = 0
    structure = ndi.generate_binary_structure(3, 1)
    for z in range(0, fg.shape[0], block_zyx[0]):
        for y in range(0, fg.shape[1], block_zyx[1]):
            for x in range(0, fg.shape[2], block_zyx[2]):
                s = (slice(z,z+block_zyx[0]), slice(y,y+block_zyx[1]), slice(x,x+block_zyx[2]))
                crop = fg[s]
                if not crop.any():
                    continue
                lab, n = ndi.label(crop, structure=structure)
                if n:
                    dst = labels[s]
                    dst[lab > 0] = lab[lab > 0] + current
                    current += n
    return labels


def audit_one(ref: np.ndarray, block: tuple[int, int, int]) -> dict:
    """Return statistics for the actual block+within-block connected-component rule."""
    if ref.ndim != 3:
        raise ValueError("ref must have shape Z,Y,X")
    sv = block_supervoxels(ref > 0, block)
    valid = ref > 0
    if not valid.any():
        raise ValueError("reference foreground is empty")
    if np.any(sv[valid] == 0) or np.any(sv[~valid] != 0):
        raise AssertionError("supervoxel foreground does not equal reference foreground")

    # Each (supervoxel ID, reference ID) pair is an intersection.
    pairs, counts = np.unique(
        np.stack([sv[valid].astype(np.int64), ref[valid].astype(np.int64)], axis=1),
        axis=0, return_counts=True,
    )
    sv_ids, positions, pair_counts = np.unique(
        pairs[:, 0], return_index=True, return_counts=True
    )
    voxel_count_per_sv = np.add.reduceat(counts, positions)
    largest_ref_overlap_per_sv = np.maximum.reduceat(counts, positions)
    mixed = pair_counts > 1
    n_ref = int(np.unique(ref[valid]).size)
    total_fg = int(valid.sum())
    minority = int((voxel_count_per_sv - largest_ref_overlap_per_sv).sum())

    # Number of different supervoxels intersecting each reference instance.
    ref_ids, supervoxels_per_ref = np.unique(pairs[:, 1], return_counts=True)
    assert len(ref_ids) == n_ref
    return {
        "block_zyx": list(block),
        "n_reference_instances": n_ref,
        "n_supervoxels": int(len(sv_ids)),
        "n_mixed_supervoxels": int(mixed.sum()),
        "fraction_mixed_supervoxels": float(mixed.mean()),
        "fraction_foreground_voxels_in_mixed_supervoxels": float(voxel_count_per_sv[mixed].sum() / total_fg),
        "minority_reference_voxel_fraction": float(minority / total_fg),
        "optimistic_majority_id_voxel_agreement": float(1 - minority / total_fg),
        "median_supervoxels_per_reference": float(np.median(supervoxels_per_ref)),
        "p90_supervoxels_per_reference": float(np.percentile(supervoxels_per_ref, 90)),
        "n_foreground_voxels": total_fg,
    }


def run(args):
    path = next(args.data_dir.glob("full_mask*.tif"), None)
    if path is None:
        raise FileNotFoundError(f"No full_mask*.tif in {args.data_dir}")
    with tifffile.TiffFile(path) as tif:
        ref_full = tif.asarray()
    if ref_full.ndim != 3:
        raise ValueError(f"Expected 3D label stack, got {ref_full.shape}")
    y, x = args.tile_y, args.tile_x
    lo, hi = args.guard, args.tile - args.guard
    if not 0 <= lo < hi or y < 0 or x < 0 or y + args.tile > ref_full.shape[1] or x + args.tile > ref_full.shape[2]:
        raise ValueError("Requested tile/guard is outside mask boundaries")
    ref = ref_full[:, y + lo:y + hi, x + lo:x + hi]

    blocks = []
    for part in args.blocks.split(","):
        elements = tuple(int(v) for v in part.lower().split("x"))
        if len(elements) != 3 or min(elements) <= 0:
            raise ValueError(f"Invalid block specification: {part!r}; expected 2x6x6")
        blocks.append(elements)
    if not blocks:
        raise ValueError("No block sizes provided")
    rows = []
    for block in blocks:
        row = audit_one(ref, block)
        rows.append(row)
        print(
            f"block={block} | sv={row['n_supervoxels']} | "
            f"mixed={row['fraction_mixed_supervoxels']:.3f} | "
            f"minority_fg={row['minority_reference_voxel_fraction']:.3f} | "
            f"median_sv_per_ref={row['median_supervoxels_per_reference']:.1f}",
            flush=True,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "supervoxel_reference_purity.csv", index=False)
    result = {
        "scope": "ONE repeatedly reused VAL crop; STRUCTURAL audit only; TEST untouched",
        "reference": "Author watershed pseudo-instances, not independent manual ground truth",
        "audit_type": "Actual grid-block plus within-block connected components, on REFERENCE foreground",
        "reference_foreground_is_oracle": True,
        "roi": {"tile_y": y, "tile_x": x, "tile_width": args.tile, "guard": args.guard, "reference_shape_ZYX": list(ref.shape)},
        "results": rows,
        "cautions": [
            "Purity measured on reference foreground; it is an optimistic check, NOT a reproduction of the predicted foreground supervoxels",
            "Minority-ID fraction is the minimum voxel assignment disagreement when each atomic supervoxel is given one reference ID; it is not PQ",
            "Connected-components supervoxels within a grid block are different from full rectangular blocks",
            "Many supervoxels per reference instance imply merging is still required, even for pure oversegmentation",
            "Do not tune a final reported model on the repeatedly reused validation crop",
            "No physical spacing or registration accuracy verified",
        ],
    }
    (args.out / "supervoxel_reference_purity.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved:", args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tile-y", type=int, default=384)
    parser.add_argument("--tile-x", type=int, default=576)
    parser.add_argument("--tile", type=int, default=192)
    parser.add_argument("--guard", type=int, default=12)
    parser.add_argument("--blocks", default="1x3x3,1x4x4,2x3x3,2x6x6", help="comma-separated ZxYxX grid sizes")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
