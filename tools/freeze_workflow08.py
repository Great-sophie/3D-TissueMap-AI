#!/usr/bin/env python3
"""Freeze Workflow08 v1.0 under releases/ without touching canonical source or results.

Default action: read-only preflight. --apply: immutable release. --verify:
validate release's own snapshot and compare current canonical code/results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tempfile

RELEASE = Path('releases/3d_spatial_organization_v1.0')
RUN = Path('runs/3d_spatial_organization_v1')
MORPH_RUN = Path('runs/3d_cell_morphometry_v1')
WORKFLOW07_RELEASE = Path('releases/3d_cell_morphometry_v1.0')
SOURCE_FILES = {
    'scripts/run_3d_spatial_organization.py': 'Official workflow08 entrypoint',
    'scripts/spatial_organization_core.py': 'Centroid neighborhood and grid-interface implementation',
    'tests/test_spatial_organization_v1.py': 'Automated spatial analysis tests',
    'docs/SPATIAL_ORGANIZATION_V1_方法与字段.md': 'Methods and field definitions',
}
TABLE_COUNT_FIELDS = {
    'instance_centroids.csv': 'n_instances',
    'instance_spatial_features.csv': 'n_instances',
    'spatial_neighbors.csv': 'n_xy_radius_pairs',
    'instance_contact_graph.csv': 'n_interface_pairs',
}
EXPECTED_FIGURES = ('xy_neighborhood_map.png', 'xy_neighbor_count_distribution.png',
                    'qc_stratified_neighborhood.png')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f'Required regular file is missing or a symlink: {path}')


def ensure_directory(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f'Expected regular directory: {path}')


def safe_relative(text: str) -> Path:
    if not isinstance(text, str) or not text or '\\' in text:
        raise RuntimeError(f'Unsafe relative path: {text!r}')
    p = PurePosixPath(text)
    if p.is_absolute() or any(part in ('', '.', '..') for part in p.parts):
        raise RuntimeError(f'Unsafe relative path: {text!r}')
    return Path(*p.parts)


def read_json(path: Path) -> dict:
    ensure_file(path)
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise RuntimeError(f'Expected JSON object: {path}')
    return result


def count_csv_rows(path: Path) -> int:
    ensure_file(path)
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        if not next(reader, None):
            raise RuntimeError(f'Missing CSV header: {path}')
        return sum(1 for row in reader if row)


def validate_expected_numbers(report: dict) -> None:
    if report.get('version') != '1.0.0':
        raise RuntimeError('Expected Workflow08 summary version 1.0.0')
    n = report['n_instances']
    if not isinstance(n, int) or n < 1:
        raise RuntimeError('Invalid n_instances')
    r, k = report['xy_radius_px'], report['radius_limited_knn_k']
    if not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0 or not isinstance(k, int) or k < 1:
        raise RuntimeError('Invalid XY radius or top-k parameter')
    get = lambda key: report[key]
    nrad, ne, nm = [get(k) for k in ('n_xy_radius_pairs', 'n_radius_limited_knn_pairs_either', 'n_radius_limited_knn_pairs_mutual')]
    pairs, xy, z = [get(k) for k in ('n_interface_pairs','n_interface_pairs_xy','n_interface_pairs_z')]
    if not (0 <= nm <= ne <= nrad) or not (max(xy,z) <= pairs <= xy+z):
        raise RuntimeError('Invalid graph-edge subset relationships')
    if report['sum_interface_faces_xy'] < xy or report['sum_interface_faces_z'] < z:
        raise RuntimeError('Interface face count smaller than pair count')
    if report.get('workflow07_interface_face_crosscheck') != 'PASS':
        raise RuntimeError('Missing successful Workflow07 interface-face crosscheck')
    strata = report['qc_strata']
    if sum(strata[g]['n_instances'] for g in ('unflagged','flagged')) != n:
        raise RuntimeError('QC strata do not add to the instance count')
    weighted = sum(strata[g]['n_instances'] * strata[g]['mean_xy_radius_neighbors'] for g in ('unflagged','flagged'))
    if not math.isclose(weighted, 2 * nrad, rel_tol=1e-8, abs_tol=1e-5):
        raise RuntimeError('XY graph degree sum disagrees with edge count')


def preflight(root: Path, check_mask: bool = True) -> dict:
    ensure_directory(root)
    ensure_directory(root / WORKFLOW07_RELEASE)
    for name in SOURCE_FILES:
        ensure_file(root / name)
    out = root / RUN
    ensure_directory(out)
    report = read_json(out / 'spatial_organization_summary.json')
    validate_expected_numbers(report)
    hashes = read_json(out / 'output_file_sha256.json')
    if not hashes:
        raise RuntimeError('Empty Workflow08 output file manifest')
    names = set(hashes)
    missing = set(TABLE_COUNT_FIELDS) | {'spatial_organization_summary.json'}
    missing -= names
    if missing:
        raise RuntimeError(f'Workflow08 manifest lacks required files: {sorted(missing)}')
    actual = {str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    expected_actual = names | {'output_file_sha256.json'}
    if actual != expected_actual:
        raise RuntimeError(f'Untracked or missing output files: {sorted(actual ^ expected_actual)}')
    for name, digest in hashes.items():
        path = out / safe_relative(name)
        ensure_file(path)
        if sha256(path) != digest:
            raise RuntimeError(f'Workflow08 output SHA256 mismatch: {name}')
    for name, field in TABLE_COUNT_FIELDS.items():
        rows = count_csv_rows(out/name)
        if rows != report[field]:
            raise RuntimeError(f'CSV row count mismatch for {name}: {rows} != {report[field]}')
    morph = root / MORPH_RUN
    for name, digest in report['workflow07_files_sha256'].items():
        path = morph / safe_relative(name)
        ensure_file(path)
        if sha256(path) != digest:
            raise RuntimeError(f'Workflow07 input modified: {name}')
    if sha256(morph / 'morphometry_summary.json') != report['workflow07_summary_sha256']:
        raise RuntimeError('Workflow07 summary SHA mismatch')
    msummary = read_json(morph / 'morphometry_summary.json')
    if msummary['source_mask_sha256'] != report['source_mask_sha256']:
        raise RuntimeError('Workflow07/08 mask provenance mismatch')
    if msummary['n_instance_ids'] != report['n_instances'] or msummary['shape_assumed_zyx'] != report['shape_assumed_zyx']:
        raise RuntimeError('Workflow07/08 instance count or shape mismatch')
    source_mask = None
    if check_mask:
        recorded = msummary.get('source_mask_path')
        if not isinstance(recorded, str) or not recorded:
            raise RuntimeError('Missing Workflow07 source mask provenance')
        # The only supported source location is within the user's own project.
        source_mask = root/'data/inspect/LVILymph'/Path(recorded).name
        ensure_file(source_mask)
        if sha256(source_mask) != report['source_mask_sha256']:
            raise RuntimeError('Original TIFF SHA-256 mismatch')
    figures = report.get('figures', [])
    if figures:
        if set(figures) != set(EXPECTED_FIGURES):
            raise RuntimeError('Unexpected or missing Workflow08 figure list')
        for file in figures:
            if f'figures/{file}' not in hashes:
                raise RuntimeError(f'Figure not included in output checksum manifest: {file}')
    return {'summary': report, 'source_hashes': {name:sha256(root/name) for name in SOURCE_FILES},
            'result_hashes': {str(RUN/name):digest for name,digest in hashes.items()},
            'output_manifest_hash': sha256(out/'output_file_sha256.json'),
            'mask_verified': bool(check_mask), 'n_result_files':len(hashes)}


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')


def documentation(summary: dict) -> str:
    strata = summary['qc_strata']
    return f'''# Workflow 08 — 3D Spatial Organization v1.0（正式研究快照）

**状态：** Exploratory spatial graph baseline. 真实运行结果与代码版本已校验，不能等同于生物学验证。

## 日常唯一代码入口

- `scripts/run_3d_spatial_organization.py` — 主运行入口
- `scripts/spatial_organization_core.py` — XY 邻域与体素界面计算核心
- `tests/test_spatial_organization_v1.py` — Workflow 08 单元和合成测试
- `docs/SPATIAL_ORGANIZATION_V1_方法与字段.md` — 指标定义
- `runs/3d_spatial_organization_v1/` — 不可随意修改的已验收输出

`releases/3d_spatial_organization_v1.0/snapshot/` 是不可变的归档副本；**不是第二套日常执行代码**。

## 真实数据摘要

| Metric | Value |
|---|---:|
| 伪实例 | {summary['n_instances']:,} |
| XY 半径（px） | {summary['xy_radius_px']} |
| 半径内 top-k | {summary['radius_limited_knn_k']} |
| 共享 Z 层 XY 邻近对 | {summary['n_xy_radius_pairs']:,} |
| radius-limited kNN either / mutual | {summary['n_radius_limited_knn_pairs_either']:,} / {summary['n_radius_limited_knn_pairs_mutual']:,} |
| Grid interface pair（union） | {summary['n_interface_pairs']:,} |
| XY / Z interface pair | {summary['n_interface_pairs_xy']:,} / {summary['n_interface_pairs_z']:,} |
| XY / Z face count | {summary['sum_interface_faces_xy']:,} / {summary['sum_interface_faces_z']:,} |
| Unflagged（平均邻居） | {strata['unflagged']['n_instances']:,} ({strata['unflagged']['mean_xy_radius_neighbors']:.3f}) |
| Flagged（平均邻居） | {strata['flagged']['n_instances']:,} ({strata['flagged']['mean_xy_radius_neighbors']:.3f}) |

## 解释边界

本研究只有一个标本，作者 watershed 产生的是自动实例 **pseudo-ID**。XY 距离仅在两个 ID 存在共同 Z 平面时、按该平面质心像素坐标计算；Z 尚未物理校准。Grid-face interfaces 不是已验证细胞接触；QC 差异主要可能受切片边界影响。没有验证细胞类型、蛋白到实例的映射或生物学相互作用。

## 维护命令

```bash
python tools/freeze_workflow08.py --verify
bash releases/3d_spatial_organization_v1.0/verify.sh
python -m pytest -q tests/test_spatial_organization_v1.py
```

新实验应使用不同的 `--out` 目录，**不要覆盖** `runs/3d_spatial_organization_v1/`。
'''


def make_release(root: Path, pre: dict) -> Path:
    destination = root / RELEASE
    if destination.exists():
        raise FileExistsError(f'Release already exists; refusing overwrite: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.wf08_release_stage_',dir=destination.parent))
    try:
        for name in SOURCE_FILES:
            dst = tmp/'snapshot'/name
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/name,dst)
        (tmp/'README_中文.md').write_text(documentation(pre['summary']),encoding='utf-8')
        with (tmp/'script_workflow_index.csv').open('w', newline='', encoding='utf-8') as f:
            writer=csv.writer(f)
            writer.writerow(['canonical_relative_path','role','sha256'])
            for name,role in SOURCE_FILES.items():
                writer.writerow([name,role,pre['source_hashes'][name]])
        report=pre['summary']
        write_json(tmp/'result_audit_summary.json', {
            'version':report['version'],
            'n_instances':report['n_instances'],
            'n_xy_radius_pairs':report['n_xy_radius_pairs'],
            'n_interface_pairs':report['n_interface_pairs'],
            'n_interface_pairs_xy':report['n_interface_pairs_xy'],
            'n_interface_pairs_z':report['n_interface_pairs_z'],
            'sum_interface_faces_xy':report['sum_interface_faces_xy'],
            'sum_interface_faces_z':report['sum_interface_faces_z'],
            'qc_strata':report['qc_strata'],
            'interpretation_limits':report.get('interpretation_limits',[]),
            'source_mask_sha256':report['source_mask_sha256'],
            'workflow07_summary_sha256':report['workflow07_summary_sha256'],
            'run_dir':str(RUN),
            'source_dir':str(WORKFLOW07_RELEASE),
        })
        (tmp/'verify.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python tools/freeze_workflow08.py --verify
python -m pytest -q tests/test_spatial_organization_v1.py
''',encoding='utf-8')
        (tmp/'run_new.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ $# -ne 1 ]]; then
  echo "Usage: bash releases/3d_spatial_organization_v1.0/run_new.sh runs/NEW_OUTPUT" >&2
  exit 2
fi
if [[ -e "$1" ]]; then
  echo "Refusing to overwrite existing output: $1" >&2
  exit 2
fi
python scripts/run_3d_spatial_organization.py \\
  --data-dir data/inspect/LVILymph \\
  --morphometry-dir runs/3d_cell_morphometry_v1 \\
  --out "$1" --assume-zyx --xy-radius-px 24 --knn-k 5
''',encoding='utf-8')
        for name in ('verify.sh','run_new.sh'):
            (tmp/name).chmod(0o755)
        lock = {
            'release_id':'3d_spatial_organization_v1.0',
            'release_type':'immutable-source-snapshot-and-live-results-manifest',
            'canonical_source_sha256':pre['source_hashes'],
            'output_sha256':pre['result_hashes'],
            'output_manifest_sha256':pre['output_manifest_hash'],
            'source_mask_sha256':report['source_mask_sha256'],
            'workflow07_summary_sha256':report['workflow07_summary_sha256'],
            'run_relpath':str(RUN),
            'unverified_spacing_notice':'ZYX is a logical axis assumption; no physical XY/Z spacing or biological cell identity verified',
        }
        write_json(tmp/'workflow08_lock.json',lock)
        files = sorted(str(f.relative_to(tmp)) for f in tmp.rglob('*') if f.is_file())
        with (tmp/'MANIFEST_SHA256.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['file','sha256'])
            for name in files:
                w.writerow([name,sha256(tmp/name)])
        validate_release_snapshot(tmp)
        if destination.exists():
            raise FileExistsError(f'Release path created concurrently: {destination}')
        tmp.rename(destination)
        return destination
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def validate_release_snapshot(release: Path) -> int:
    ensure_directory(release)
    manifest = release/'MANIFEST_SHA256.csv'
    ensure_file(manifest)
    with manifest.open('r',newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['file','sha256']:
            raise RuntimeError('Malformed release SHA manifest')
        entries=list(reader)
    if not entries:
        raise RuntimeError('Empty release SHA manifest')
    names=set()
    for row in entries:
        rel=safe_relative(row['file'])
        if str(rel) in names:
            raise RuntimeError('Duplicate release manifest entry')
        names.add(str(rel))
        p=release/rel
        ensure_file(p)
        if any(parent.is_symlink() for parent in list(p.parents) if parent!=release and release in parent.parents):
            raise RuntimeError(f'Symlinked release directory: {p}')
        if sha256(p)!=row['sha256']:
            raise RuntimeError(f'Release snapshot SHA mismatch: {rel}')
    actual={str(p.relative_to(release)) for p in release.rglob('*') if p.is_file()}
    if actual != names|{'MANIFEST_SHA256.csv'}:
        raise RuntimeError(f'Untracked/missing release files: {sorted(actual ^ (names | {"MANIFEST_SHA256.csv"}))}')
    return len(entries)


def verify(root: Path, check_live: bool = True, check_mask: bool = True) -> dict:
    destination = root/RELEASE
    n=validate_release_snapshot(destination)
    lock=read_json(destination/'workflow08_lock.json')
    if lock.get('release_id')!='3d_spatial_organization_v1.0':
        raise RuntimeError('Wrong release lock identity')
    for name, expected in lock['canonical_source_sha256'].items():
        path=destination/'snapshot'/safe_relative(name)
        ensure_file(path)
        if sha256(path)!=expected:
            raise RuntimeError(f'Frozen script does not match lock: {name}')
    if check_live:
        pre=preflight(root,check_mask)
        if pre['source_hashes'] != lock['canonical_source_sha256']:
            raise RuntimeError('Live Workflow08 source differs from frozen release')
        if pre['result_hashes'] != lock['output_sha256']:
            raise RuntimeError('Live Workflow08 output differs from frozen release')
        if pre['output_manifest_hash'] != lock['output_manifest_sha256']:
            raise RuntimeError('Live output checksum manifest differs from frozen release')
        if pre['summary']['source_mask_sha256'] != lock['source_mask_sha256']:
            raise RuntimeError('Source mask provenance changed')
        if pre['summary']['workflow07_summary_sha256'] != lock['workflow07_summary_sha256']:
            raise RuntimeError('Workflow07 summary provenance changed')
    return {'release_files_verified':n, 'checked_live':bool(check_live), 'release':str(destination)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--apply',action='store_true',help='Create release once (never overwrite)')
    group.add_argument('--verify',action='store_true',help='Check release and all live canonical files')
    group.add_argument('--verify-snapshot-only',action='store_true',help='Verify archival snapshot even when live files evolve')
    args=parser.parse_args(argv)
    root=args.project_root.resolve()
    if args.verify or args.verify_snapshot_only:
        result=verify(root,check_live=args.verify)
        print('PASS: Workflow08 release SHA-256:',json.dumps(result,ensure_ascii=False))
        return 0
    if (root/RELEASE).exists():
        raise FileExistsError(f'Release already exists: {root/RELEASE}; use --verify')
    pre=preflight(root,check_mask=True)
    report=pre['summary']
    print('PASS: Workflow08 preflight —',json.dumps({
        'n_instances':report['n_instances'],
        'n_xy_radius_pairs':report['n_xy_radius_pairs'],
        'n_interface_pairs':report['n_interface_pairs'],
        'n_output_files_verified':pre['n_result_files'],
        'mask_sha256_verified':pre['mask_verified'],
    },ensure_ascii=False))
    print('Source snapshot entries:',', '.join(SOURCE_FILES))
    print('Destination:',root/RELEASE)
    if args.apply:
        dest=make_release(root,pre)
        print('FROZEN:',dest)
        print('Next: python tools/freeze_workflow08.py --verify')
    else:
        print('DRY RUN ONLY: add --apply after reviewing results')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
