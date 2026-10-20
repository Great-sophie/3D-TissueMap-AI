#!/usr/bin/env python3
"""Run Workflow 07 / 3D Cell Morphometry v1.0 on author pseudo-ID TIFF."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json
import tifffile

from cell_morphometry_core import save_tables, validate_mask


def choose_mask(data_dir: Path | None, mask_path: Path | None) -> Path:
    if mask_path is not None:
        selected = mask_path.expanduser()
        if not selected.is_file():
            raise FileNotFoundError(f"Cannot find --mask: {selected}")
        return selected
    if data_dir is None:
        raise ValueError("Specify either --data-dir or --mask")
    options = sorted(data_dir.glob("full_mask*.tif")) + sorted(data_dir.glob("full_mask*.tiff"))
    if len(options) != 1:
        raise ValueError(f"Expected exactly one full_mask*.tif(f) in {data_dir}; found {len(options)}. "
                         "Specify --mask for an unambiguous input.")
    return options[0]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", type=Path, help="Directory containing one full_mask*.tif(f)")
    group.add_argument("--mask", type=Path, help="Exact path to author 3D instance-ID TIFF")
    p.add_argument("--out", type=Path, default=Path("runs/3d_cell_morphometry_v1"))
    p.add_argument("--assume-zyx", action="store_true", required=False,
                   help="Required acknowledgment: mask stack treated as Z,Y,X (not proven physical order)")
    p.add_argument("--min-qc-voxels", type=int, default=12,
                   help="Heuristic flag only; NOT a biological threshold (default 12)")
    p.add_argument("--no-figures", action="store_true", help="Skip PNG figures")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.assume_zyx:
        raise SystemExit("Refusing to infer stack axes. Add --assume-zyx after checking that input is intended Z,Y,X.")
    path = choose_mask(args.data_dir, args.mask)
    if args.out.exists():
        raise SystemExit(f"Output already exists: {args.out}. Choose a new --out; do not overwrite established results.")
    print("Input:",path)
    print("Assumption: axes Z,Y,X; voxel/pixel units ONLY; pseudo-reference watershed IDs.")
    with tifffile.TiffFile(path) as tf:
        if len(tf.series) != 1:
            raise ValueError(f"Expected exactly 1 TIFF series, got {len(tf.series)}")
        axes = str(tf.series[0].axes)
        shape = tuple(int(i) for i in tf.series[0].shape)
        print("Reported TIFF series axes:",axes, "shape:",shape,
              "(TIFF axis letters do not validate physical registration)")
        mask = tf.asarray()
    validate_mask(mask)
    result = save_tables(mask,path,args.out,min_qc_voxels=args.min_qc_voxels,
                         figures=not args.no_figures,tiff_series_axes=axes)
    print(json.dumps({key:result[key] for key in
       ("shape_assumed_zyx", "n_instance_ids", "foreground_voxels",
        "n_multiplane_ids", "n_z_gapped_ids", "n_multicomponent_6",
        "n_qc_review_flagged")},ensure_ascii=False,indent=2))
    print("Saved:",args.out)
    print("CAUTION: no µm/µm²/µm³, no protein join, no physically verified z-spacing.")
    return result


if __name__ == "__main__":
    main()
