#!/usr/bin/env python3
"""Prepare five audited registered IMC channels from author ZIP for Workflow09C.

Source archives are read-only. This does not establish image-to-mask registration or
whole-cell protein expression; it only builds reproducible exploratory inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import tifffile

MARKERS = {'Ir193': 'Ir193', 'panCK': 'Cd112', 'HER2': 'Gd156', 'CD31': 'Eu151', 'LYVE1': 'Er168'}
VERSION = '1.0.0'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def basekey(name: str) -> str:
    s = Path(name).name
    s = re.sub(r'^\d{4}_', '', s)
    s = re.sub(r'__[A-Za-z0-9]+\.tiff?$', '', s, flags=re.I)
    s = re.sub(r'\.ome\.tiff?$', '', s, flags=re.I)
    return s


def read_order(z: ZipFile, depth: int) -> dict[int, str]:
    matches = [n for n in z.namelist() if Path(n).name == 'final_3D_stack_order.csv']
    if len(matches) != 1:
        raise ValueError('Need exactly one final_3D_stack_order.csv in model ZIP')
    rows = list(csv.reader(StringIO(z.read(matches[0]).decode('utf-8-sig'))))
    mapping: dict[int, str] = {}
    for row in rows:
        if len(row) < 2:
            raise ValueError(f'Invalid stack-order CSV row: {row}')
        k = int(row[1])
        if k in mapping:
            raise ValueError(f'Duplicate stack Z index {k}')
        mapping[k] = basekey(row[0])
    if set(mapping) != set(range(depth)):
        raise ValueError(f'Expected zero-based stack order 0..{depth-1}, found {sorted(mapping)}')
    return mapping


def _zip_marker_members(z: ZipFile, depth: int, metal: str, order: dict[int, str]):
    names = [n for n in z.namelist() if f'/{metal}/' in n and re.search(r'\.tiff?$', n, re.I)]
    if len(names) != depth:
        raise ValueError(f'{metal}: expected {depth} TIFFs, found {len(names)}')
    indexed: dict[int, str] = {}
    for name in names:
        if not name.endswith(('.tiff', '.tif')):
            raise ValueError(f'{metal}: unexpected extension {name}')
        if not re.search(rf'__{metal}\.tiff?$', Path(name).name, re.I):
            raise ValueError(f'{metal}: filename suffix does not match isotope: {name}')
        m = re.match(r'^(\d{4})_', Path(name).name)
        if m is None:
            raise ValueError(f'{metal}: missing numbered Z prefix: {name}')
        idx = int(m.group(1)) - 1
        if idx in indexed:
            raise ValueError(f'{metal}: repeated Z={idx}')
        if order.get(idx) != basekey(name):
            raise ValueError(f'{metal}: Z={idx} conflicts with author stack order: {name}')
        indexed[idx] = name
    if set(indexed) != set(range(depth)):
        raise ValueError(f'{metal}: missing numbered slice')
    return [indexed[z] for z in range(depth)]


def inspect(mask_path: Path, image_zip: Path, model_zip: Path) -> dict:
    for p in (mask_path, image_zip, model_zip):
        if not p.is_file():
            raise FileNotFoundError(p)
    with tifffile.TiffFile(mask_path) as tif:
        if len(tif.series) != 1:
            raise ValueError('Full mask must have one TIFF series')
        shape = tuple(map(int, tif.series[0].shape))
        dtype = np.dtype(tif.series[0].dtype)
    if len(shape) != 3 or not np.issubdtype(dtype, np.integer):
        raise ValueError(f'Full mask must be 3D integer instance image, got {shape} {dtype}')
    depth, height, width = shape
    with ZipFile(model_zip) as mz, ZipFile(image_zip) as iz:
        order = read_order(mz, depth)
        inventories = {}
        for marker, metal in MARKERS.items():
            names = _zip_marker_members(iz, depth, metal, order)
            checked = []
            for z, name in enumerate(names):
                inf = iz.getinfo(name)
                # Read all selected TIFFs; ZipFile.read checks the ZIP CRC.
                raw = iz.read(name)
                with tifffile.TiffFile(BytesIO(raw)) as tif:
                    if len(tif.series) != 1:
                        raise ValueError(f'{marker} z{z}: multiple TIFF series')
                    series = tif.series[0]
                    found_shape = tuple(map(int, series.shape))
                    found_dtype = str(series.dtype)
                    axes = series.axes
                if found_shape != (height, width) or axes != 'YX':
                    raise ValueError(f'{marker} z{z}: shape/axes {found_shape}/{axes}, expected {(height, width)}/YX')
                if not np.issubdtype(np.dtype(found_dtype), np.number):
                    raise ValueError(f'{marker} z{z}: nonnumeric dtype {found_dtype}')
                checked.append({'z_index': z, 'member': name, 'filename': Path(name).name,
                                'shape': list(found_shape), 'axes': axes, 'dtype': found_dtype,
                                'zip_crc32': f'{inf.CRC:08x}', 'uncompressed_bytes': inf.file_size,
                                'sha256': hashlib.sha256(raw).hexdigest()})
            inventories[marker] = {'metal': metal, 'slices': checked}
    return {'version': VERSION, 'shape_assumed_zyx': list(shape),
            'n_markers': len(inventories), 'n_selected_tiffs': sum(len(v['slices']) for v in inventories.values()),
            'markers': inventories,
            'limitations': ['Author pseudo-instance mask, not manually verified cell identity',
                            'Matching TIFF dimensions and stack order do not independently prove registration',
                            'Mask-restricted marker intensity is not whole-cell protein expression']}


def _manifest(mask: Path, data_out: Path, audit: dict) -> dict:
    specs = []
    for marker, record in audit['markers'].items():
        files = [str((data_out / marker / s['filename']).resolve()) for s in record['slices']]
        specs.append({'name': marker, 'mode': 'per_section', 'axes': 'YX', 'files': files})
    return {'schema_version': 1, 'mask_path': str(mask.resolve()),
            'expected_mask_sha256': sha256(mask),
            'image_coordinate_status': 'unverified_assumed_same_index_grid', 'markers': specs}


def prepare(mask: Path, image_zip: Path, model_zip: Path, data_out: Path, manifest_out: Path, apply: bool):
    audit = inspect(mask, image_zip, model_zip)
    manifest = _manifest(mask, data_out, audit)
    print(f"CHECK: {audit['n_markers']} mapped markers, {audit['n_selected_tiffs']} TIFFs, shape {audit['shape_assumed_zyx']}")
    print('Markers:', ', '.join(f'{k}={v["metal"]}' for k,v in audit['markers'].items()))
    print('Output data:', data_out)
    print('Output manifest:', manifest_out)
    if not apply:
        print('DRY-RUN: nothing written; rerun with --apply')
        return audit
    if data_out.exists() or manifest_out.exists():
        raise FileExistsError('Destination exists; never overwrite prepared marker data or manifest')
    data_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.wf09c_prepare_', dir=data_out.parent))
    published = False
    try:
        with ZipFile(image_zip) as iz:
            for marker, record in audit['markers'].items():
                d = stage / marker
                d.mkdir()
                for item in record['slices']:
                    dest = d / item['filename']
                    dest.write_bytes(iz.read(item['member']))
                    if sha256(dest) != item['sha256']:
                        raise ValueError(f'Extracted marker file SHA mismatch: {dest}')
        audit['source_zip_sha256'] = sha256(image_zip)
        audit['model_zip_sha256'] = sha256(model_zip)
        audit['full_mask_sha256'] = manifest['expected_mask_sha256']
        (stage / 'source_inventory.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        stage.rename(data_out)
        published = True
        tmpmanifest = manifest_out.with_name(manifest_out.name+'.staging')
        if tmpmanifest.exists():
            raise FileExistsError(tmpmanifest)
        try:
            tmpmanifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            tmpmanifest.rename(manifest_out)
        finally:
            if tmpmanifest.exists():
                tmpmanifest.unlink()
    except Exception:
        # Roll back only our own newly-created published data on failure.
        if published and not manifest_out.exists() and data_out.exists():
            shutil.rmtree(data_out)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print('PREPARED:', len(audit['markers']), 'markers; manifest:', manifest_out)
    return audit


def verify(mask: Path, image_zip: Path, model_zip: Path, data_out: Path, manifest_out: Path):
    if not manifest_out.is_file() or not data_out.is_dir():
        raise FileNotFoundError('Prepared manifest or data directory missing')
    audit = json.loads((data_out / 'source_inventory.json').read_text())
    manifest = json.loads(manifest_out.read_text())
    if audit['full_mask_sha256'] != sha256(mask):
        raise ValueError('Input mask changed')
    if audit['source_zip_sha256'] != sha256(image_zip) or audit['model_zip_sha256'] != sha256(model_zip):
        raise ValueError('Source ZIP changed')
    if manifest != _manifest(mask, data_out, audit):
        raise ValueError('Manifest changed or paths differ')
    n = 0
    for marker, record in audit['markers'].items():
        for item in record['slices']:
            p = data_out / marker / item['filename']
            if not p.is_file() or sha256(p) != item['sha256']:
                raise ValueError(f'Marker file missing or changed: {p}')
            n += 1
    expected_files = {str((data_out / m / s['filename']).resolve())
                      for m,r in audit['markers'].items() for s in r['slices']}
    if {p for spec in manifest['markers'] for p in spec['files']} != expected_files:
        raise ValueError('Manifest marker member mismatch')
    print(f'PASS: {n} marker TIFFs and manifest checksums match source inventory')
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--check', action='store_true')
    grp.add_argument('--apply', action='store_true')
    grp.add_argument('--verify', action='store_true')
    ap.add_argument('--zip', type=Path, default=Path('data/inspect/LVILymph/LVILymphBreastCancer_SIMILARITY10_In115.zip'))
    ap.add_argument('--model-zip', type=Path, default=Path('data/raw/LVILymphBreastCancerModel.zip'))
    ap.add_argument('--mask', type=Path, default=Path('data/inspect/LVILymph/full_mask_final_segmentation_hwatershed_bg500_90%.tif'))
    ap.add_argument('--out-data', type=Path, default=Path('data/processed/workflow09_five_markers'))
    ap.add_argument('--manifest', type=Path, default=Path('configs/protein_marker_manifest_registered_5channels.json'))
    a = ap.parse_args(argv)
    if a.verify:
        verify(a.mask,a.zip,a.model_zip,a.out_data,a.manifest)
    else:
        prepare(a.mask,a.zip,a.model_zip,a.out_data,a.manifest,a.apply)

if __name__ == '__main__':
    main()
