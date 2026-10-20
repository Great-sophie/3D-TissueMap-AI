#!/usr/bin/env python3
"""Read-only audit of Cell Tracking Challenge 3D+Time training archives.

Expected CTC names: 01/t000.tif, 01_GT/SEG/man_seg000.tif.
This program deliberately does NOT treat background voxels as negative GT in real CTC.
"""
import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import tifffile

RAW_RE = re.compile(r"(?:^|/)(?P<seq>\d{2})/t(?P<time>\d+)\.tiff?$", re.I)
GT_RE = re.compile(r"(?:^|/)(?P<seq>\d{2})_GT/SEG/man_seg(?P<time>\d+)\.tiff?$", re.I)


def scan_archive(filename: Path, max_pairs: int = 3):
    if not filename.is_file():
        raise FileNotFoundError(filename)
    if max_pairs < 1:
        raise ValueError("--max-pairs must be >=1")
    with zipfile.ZipFile(filename) as zf:
        raw, gt = {}, {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace('\\', '/')
            m = RAW_RE.search(name)
            if m:
                k = (m.group('seq'), int(m.group('time')))
                if k in raw:
                    raise ValueError(f"Duplicate raw frame for {k}")
                raw[k] = name
            m = GT_RE.search(name)
            if m:
                k = (m.group('seq'), int(m.group('time')))
                if k in gt:
                    raise ValueError(f"Duplicate GT frame for {k}")
                gt[k] = name
        pairs = sorted(set(raw) & set(gt))
        summary = {
            'zip_path': str(filename.resolve()),
            'zip_size_bytes': filename.stat().st_size,
            'raw_frame_count': len(raw),
            'seg_gt_frame_count': len(gt),
            'matched_frame_count': len(pairs),
            'raw_sequences': sorted({k[0] for k in raw}),
            'gt_sequences': sorted({k[0] for k in gt}),
            'note': 'CTC real-data gold segmentation is sparsely annotated; zero voxels are NOT guaranteed to mean background.',
            'audited_pairs': [],
        }
        if not raw:
            raise ValueError("No raw frames (e.g. 01/t000.tif) in CTC archive")
        if not gt:
            raise ValueError("No gold segmentation masks (e.g. 01_GT/SEG/man_seg000.tif)")
        if not pairs:
            raise ValueError("Raw frames and SEG references do not match by sequence/time")
        for seq, t in pairs[:max_pairs]:
            rname = raw[(seq, t)]
            gname = gt[(seq, t)]
            with zf.open(rname) as f:
                image = tifffile.imread(io.BytesIO(f.read()))
            with zf.open(gname) as f:
                label = tifffile.imread(io.BytesIO(f.read()))
            if image.ndim != 3 or label.ndim != 3:
                raise ValueError(f"Expected 3D ZYX frames; {rname}: {image.shape}, {gname}: {label.shape}")
            if image.shape != label.shape:
                raise ValueError(f"Raw/GT shape mismatch for {seq} t={t}: {image.shape} vs {label.shape}")
            if not np.issubdtype(label.dtype, np.integer):
                raise ValueError(f"GT must have integer instance IDs: {gname}: {label.dtype}")
            ids = np.unique(label)
            ids = ids[ids > 0]
            spanning = 0
            if ids.size:
                # Avoid constructing a giant (instances, Z, Y, X) Boolean array.
                per_z_ids = [np.unique(label[z, :, :]) for z in range(label.shape[0])]
                seen_ids, z_counts = np.unique(np.concatenate(per_z_ids), return_counts=True)
                spanning = int(np.count_nonzero((seen_ids > 0) & (z_counts >= 2)))
            summary['audited_pairs'].append({
                'sequence': seq, 'time': t,
                'raw_file': rname, 'gt_file': gname,
                'shape_zyx': list(image.shape),
                'raw_dtype': str(image.dtype), 'gt_dtype': str(label.dtype),
                'n_3d_instance_ids': int(ids.size),
                'ids_spanning_at_least_2_z_planes': spanning,
                'gt_nonzero_voxel_fraction': float(np.count_nonzero(label) / label.size),
            })
    return summary


def main():
    p = argparse.ArgumentParser(description='Audit paired volumetric frames and whole-cell GT in a CTC training ZIP without extraction.')
    p.add_argument('--zip', required=True, type=Path)
    p.add_argument('--max-pairs', default=3, type=int)
    p.add_argument('--output', type=Path, help='Optional JSON summary path')
    a = p.parse_args()
    try:
        s = scan_archive(a.zip, a.max_pairs)
        result = {'status':'PASS', **s}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if a.output:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f'AUDIT FAILED: {type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
