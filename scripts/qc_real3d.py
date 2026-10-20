#!/usr/bin/env python3
"""Inspect Kuett 3D IMC LVILymph archive without treating shape agreement as registration proof.

Run from the project root:
    python qc_real3d.py --data-dir data/inspect/LVILymph \
        --out runs/lvilymph_real_qc

Dependencies: numpy, pandas, tifffile, matplotlib (all in tempodino, if installed).
"""
import argparse
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def normalize(a):
    arr = np.asarray(a, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, (1, 99.5))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((np.nan_to_num(arr, nan=float(lo)) - lo) / (hi - lo), 0, 1)


def parse_index(filename):
    m = re.match(r'^(\d+)_', PurePosixPath(filename).name)
    return int(m.group(1)) if m else None


def build_index(zfile):
    grouped = defaultdict(list)
    for info in zfile.infolist():
        if info.is_dir() or not info.filename.lower().endswith(('.tif', '.tiff')):
            continue
        parts = PurePosixPath(info.filename).parts
        if len(parts) < 2:
            continue
        grouped[parts[-2]].append((parse_index(info.filename), info.filename))
    for name, entries in grouped.items():
        grouped[name] = sorted(entries, key=lambda x: (x[0] is None, x[0] or 0, x[1]))
    return grouped


def read_zip_tif(zfile, name):
    with zfile.open(name) as f:
        im = tifffile.imread(io.BytesIO(f.read()))
    if im.ndim != 2:
        raise ValueError(f'Expected 2D TIFF but got {im.shape} for {name}')
    return im


def mask_report(mask, label_ids):
    unique = np.unique(mask)
    ids = unique[unique != 0]
    adjacent = []
    for zi in range(len(mask) - 1):
        a, b = mask[zi], mask[zi + 1]
        shared_fg = (a != 0) & (b != 0)
        cnt = int(shared_fg.sum())
        adjacent.append({
            'z_pair': [zi, zi + 1],
            'shared_foreground_pixels': cnt,
            'same_label_pixel_fraction_on_shared_foreground':
                float(np.count_nonzero(shared_fg & (a == b)) / cnt) if cnt else None,
            'shared_label_id_count': int(len((set(np.unique(a)) & set(np.unique(b))) - {0})),
        })
    present = set(int(i) for i in ids)
    area = set(int(i) for i in label_ids)
    return {
        'shape': list(mask.shape), 'dtype': str(mask.dtype),
        'unique_nonzero_label_ids': len(present),
        'nonzero_voxel_fraction': float(np.mean(mask != 0)),
        'label_ids_in_area_csv': len(area),
        'area_csv_ids_found_in_full_mask': len(area & present),
        'area_csv_ids_not_found_sample': sorted(area - present)[:15],
        'neighboring_slices_label_persistence': adjacent,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=Path('data/inspect/LVILymph'))
    ap.add_argument('--out', type=Path, default=Path('runs/lvilymph_real_qc'))
    ap.add_argument('--slice', type=int, default=7, help='Zero-based plane, default 7')
    ap.add_argument('--markers', default='Ir193,panCK,LYVE1,CD31', help='Comma-separated clean_target or channel_target names')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = args.data_dir
    full_files = list(d.glob('full_mask*.tif'))
    measure_files = list(d.glob('measured_mask*.tif'))
    zipfiles = list(d.glob('*SIMILARITY*.zip'))
    if len(full_files) != 1 or len(zipfiles) != 1:
        raise FileNotFoundError(f'Expected one full mask and one SIMILARITY ZIP inside {d}')

    full = tifffile.imread(full_files[0])
    if full.ndim != 3 or full.dtype.kind not in 'ui':
        raise ValueError(f'Expected 3D integer label mask; got shape {full.shape}, dtype {full.dtype}')
    zcount = full.shape[0]
    k = args.slice
    if not (0 <= k < zcount):
        raise ValueError(f'--slice must be between 0 and {zcount - 1}')

    areas_path = d / 'LVI_lymph_labels_area.csv'
    area = pd.read_csv(areas_path)
    if 'label' not in area.columns:
        raise ValueError('Area CSV lacks expected label column')
    rep = {'source': 'REAL Kuett LVILymph data, user-downloaded',
           'registered_stack_status': 'UNVERIFIED: filenames/shape do not prove correspondence',
           'physical_voxel_spacing_um': 'UNVERIFIED: no physical lengths calculated',
           'full_mask': mask_report(full, area['label'].dropna().astype(int).tolist())}
    if measure_files:
        with tifffile.TiffFile(measure_files[0]) as t:
            rep['measured_mask'] = {'shape': list(t.series[0].shape),
                                    'axes': t.series[0].axes,
                                    'same_xy_dimensions_as_full': list(t.series[0].shape[-2:]) == list(full.shape[-2:])}

    panel = pd.read_csv(d / 'LVI_lymph_panel.csv')
    intensities = pd.read_csv(d / 'LVI_lymph_mean_intensities.csv', nrows=1)
    n_int = sum(1 for _ in open(d / 'LVI_lymph_mean_intensities.csv', encoding='utf-8')) - 1
    rep['single_cell_tables'] = {
        'mean_intensity_rows': n_int,
        'labels_area_rows': len(area),
        'mean_intensity_has_label_column': 'label' in intensities.columns,
        'label_to_intensity_row_mapping_verified': False,
        'warning': 'NEVER join protein intensities to labels by row position without checking source preprocessing code'
    }
    orderpath = d / 'final_3D_stack_order.csv'
    if orderpath.exists():
        order_df = pd.read_csv(orderpath)
        rep['stack_order_csv'] = {'columns': list(order_df.columns), 'rows': order_df.astype(str).to_dict(orient='records')}

    markers = [x.strip() for x in args.markers.split(',') if x.strip()]
    frames = {}
    with ZipFile(zipfiles[0]) as z:
        idx = build_index(z)
        rep['image_zip'] = {
            'archive': zipfiles[0].name,
            'channel_count': len(idx),
            'channel_directories': {n: {'count': len(v), 'slice_indices': [s for s, _ in v]} for n, v in sorted(idx.items())},
        }
        table = {}
        for _, row in panel.iterrows():
            for col in ('clean_target', 'channel_target', 'Target'):
                val = row.get(col)
                if isinstance(val, str) and val.strip():
                    table[val.strip().lower()] = str(row['Metal Tag']).strip()
        for marker in markers:
            metal = table.get(marker.lower(), marker)
            if metal not in idx:
                print(f'WARNING: requested {marker} maps to {metal}, absent in ZIP channel dirs')
                continue
            entries = idx[metal]
            numbers = [x for x, _ in entries]
            if len(entries) != zcount or len(set(numbers)) != zcount or None in numbers:
                print(f'WARNING: {marker} has invalid slice count/numbering {numbers}; skipping display')
                continue
            mip = None
            center = None
            dtype = None
            for i, (_, name) in enumerate(entries):
                img = read_zip_tif(z, name)
                if img.shape != full.shape[1:]:
                    raise ValueError(f'{marker} image shape {img.shape} differs from full_mask {full.shape[1:]}')
                if mip is None:
                    mip = img.copy()
                else:
                    np.maximum(mip, img, out=mip)
                if i == k:
                    center = img.copy()
                dtype = str(img.dtype)
            frames[marker] = (center, mip)
            rep.setdefault('selected_channels', {})[marker] = {
                'metal_tag_folder': metal, 'frames': len(entries), 'spatial_shape_matched': True,
                'first_slice_index': numbers[0], 'last_slice_index': numbers[-1],
                'dtype': dtype, 'alignment_with_mask_verified': False,
            }
            print('OK', marker, metal, f'{len(entries)} slices, shape=', center.shape, 'dtype=', dtype)

    if not frames:
        raise ValueError('None of the requested channels could be inspected')

    fig, axes = plt.subplots(2, len(frames), figsize=(4.5 * len(frames), 8), squeeze=False)
    for col, (marker, (center, mip)) in enumerate(frames.items()):
        axes[0, col].imshow(normalize(center), cmap='gray')
        axes[0, col].set_title(f'{marker}  slice {k + 1}/{zcount}')
        axes[1, col].imshow(normalize(mip), cmap='gray')
        axes[1, col].set_title(f'{marker}  XY max projection')
        for r in (0, 1):
            axes[r, col].axis('off')
    fig.suptitle('REAL 3D IMC | image order and mask registration not yet validated')
    fig.tight_layout()
    fig.savefig(args.out / 'real_markers_xy.png', dpi=125)
    plt.close(fig)

    # This overlay is a VISUAL SCREEN only; it is not proof of registration.
    if 'Ir193' in frames:
        img = normalize(frames['Ir193'][0])
    else:
        img = normalize(next(iter(frames.values()))[0])
    rgb = np.repeat(img[..., None], 3, axis=2)
    m = full[k]
    edge = (m != 0) & ((m != np.roll(m, 1, axis=0)) | (m != np.roll(m, 1, axis=1)))
    edge[:1, :] = False
    edge[:, :1] = False
    rgb[edge] = [1.0, 0.2, 0.1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(rgb)
    ax.set_title(f'Provisional nuclear/mask overlay — plane index {k} — VERIFY order/registration')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(args.out / 'mask_overlay_provisional.png', dpi=140)
    plt.close(fig)

    rep['interpretation_cautions'] = [
        'Mask TIFF QYX is an unnamed first axis; do not infer Z orientation from axes string alone.',
        'Z slice order in images is numeric filename order; final_3D_stack_order.csv must be reconciled with this before 3D analysis.',
        'Matching spatial dimensions are necessary, NOT sufficient, to establish registration.',
        'Mean-intensities CSV lacks an explicit label ID; do not join by row index without external evidence.',
        'Do not use provided physical Z/XY spacings from a different dataset or synthetic demonstration.',
        'Published reference watershed masks are algorithm-derived, not automatically manual ground truth.',
    ]
    with (args.out / 'real_qc.json').open('w', encoding='utf-8') as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print('\nSaved:')
    for name in ('real_markers_xy.png', 'mask_overlay_provisional.png', 'real_qc.json'):
        print(args.out / name)
    print('\nMask IDs:', rep['full_mask']['unique_nonzero_label_ids'])
    print('Table rows:', rep['single_cell_tables'])
    print('REPORT STATUS: geometric & ID correspondence is not yet verified. No micron-scale results claimed.')

if __name__ == '__main__':
    main()
