#!/usr/bin/env python3
"""Freeze Workflow 10 offline 3D viewer v1.0 with original HTML and provenance.

Run at project root. Default is read-only preflight; --apply creates a release
atomically without overwriting anything, --verify checks both archive and live
sources. No GPU/Plotly required to freeze an already generated viewer.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

REL = Path('releases/3d_tissue_microenvironment_visualization_v1.0')
PREVIOUS_REL = Path('releases/3d_protein_marker_integration_v1.0')
VIEWER = Path('runs/3d_tissue_microenvironment_viewer_v1')
QC = Path('runs/3d_her2_hotspot_qc_v1')
MASK = Path('data/inspect/LVILymph/full_mask_final_segmentation_hwatershed_bg500_90%.tif')
SPATIAL_TABLE = Path('runs/3d_protein_marker_integration_v1/instance_marker_spatial_table.csv')
PROTEIN_SUMMARY = Path('runs/3d_protein_marker_integration_v1/marker_integration_summary.json')
MORPH_TABLE = Path('runs/3d_cell_morphometry_v1/cell_morphometry.csv')
SOURCE_REQUIRED = ['scripts/build_3d_tissue_viewer.py',
                   'tools/freeze_workflow10.py',
                   'tests/test_freeze_workflow10.py']
SOURCE_OPTIONAL = ['tests/test_workflow09_finalize_and_visualize.py']
VIEWER_OUTPUTS = ['centroid_marker_viewer.html','roi_voxel_viewer.html','viewer_summary.json']
QC_OUTPUTS = ['her2_hotspot_blocks_120px.csv','her2_core_roi_registered_overlay.png']
MARKERS = {'Ir193','panCK','HER2','CD31','LYVE1'}
HEX64 = re.compile(r'^[0-9a-f]{64}$')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f'Required ordinary file unavailable: {path}')


def read_json(path: Path):
    regular(path)
    return json.loads(path.read_text(encoding='utf-8'))


def files_hash(folder: Path, exclude: str) -> dict:
    if not folder.is_dir() or folder.is_symlink():
        raise RuntimeError(f'Archive folder missing: {folder}')
    files = sorted(folder.rglob('*'))
    if any(p.is_symlink() for p in files):
        raise RuntimeError(f'Symlink found inside archive: {folder}')
    return {p.relative_to(folder).as_posix(): sha(p) for p in files
            if p.is_file() and p.relative_to(folder).as_posix() != exclude}


def verify_previous_release(root: Path) -> dict:
    release = root / PREVIOUS_REL
    lock = read_json(release / 'provenance_lock.json')
    manifest = read_json(release / 'release_sha256.json')
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError('Workflow09 frozen manifest missing or empty')
    if manifest != files_hash(release, 'release_sha256.json'):
        raise RuntimeError('Workflow09 release archive changed; run 09 verification')
    if lock.get('n_instances') != 12663 or lock.get('n_markers') != 5:
        raise RuntimeError('Workflow09 release is not the expected frozen baseline')
    return lock


def verify_viewer_output(root: Path, n_expected: int) -> dict:
    folder = root / VIEWER
    manifest = read_json(folder / 'viewer_sha256.json')
    if not isinstance(manifest, dict) or set(manifest) != set(VIEWER_OUTPUTS):
        raise RuntimeError('Unexpected viewer_sha256.json entry list')
    actual = files_hash(folder, 'viewer_sha256.json')
    if manifest != actual:
        raise RuntimeError('Viewer SHA-256 manifest mismatch; regenerate/check outputs')
    summary = read_json(folder / 'viewer_summary.json')
    if summary.get('n_instances') != n_expected:
        raise RuntimeError('Viewer instance count changed')
    if summary.get('z_units') != 'integer slice index ONLY; physical spacing unverified':
        raise RuntimeError('Viewer Z geometry labeling changed')
    if not isinstance(summary.get('roi'), dict) or summary['roi'].get('visualized_voxels',0) < 1:
        raise RuntimeError('Viewer ROI metadata missing')
    for name in VIEWER_OUTPUTS[:2]:
        path = folder / name
        with path.open('rb') as f:
            intro = f.read(300)
        if b'<html' not in intro.lower() and b'<!doctype html' not in intro.lower():
            raise RuntimeError('Viewer HTML not recognizable: ' + name)
        if path.stat().st_size < 1000:
            raise RuntimeError('Viewer HTML unexpectedly tiny: '+name)
    return summary


def verify_hotspot_qc(root: Path) -> dict:
    folder = root / QC
    for name in QC_OUTPUTS:
        regular(folder / name)
    csvfile = folder / QC_OUTPUTS[0]
    with csvfile.open(newline='',encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'block_x','block_y','n_instances','n_her2_high','her2_high_fraction'}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError('HER2 hotspot CSV missing expected columns')
        if not any(True for _ in reader):
            raise RuntimeError('HER2 hotspot CSV is empty')
    with (folder/QC_OUTPUTS[1]).open('rb') as f:
        if f.read(8)!=b'\x89PNG\r\n\x1a\n':
            raise RuntimeError('HER2 overlay is not a PNG')
    return {(QC / name).as_posix():sha(folder/name) for name in QC_OUTPUTS}


def preflight(root: Path) -> dict:
    if not root.is_dir():
        raise RuntimeError('Project root missing')
    source = {}
    for rel in SOURCE_REQUIRED:
        regular(root/rel)
        source[rel] = sha(root/rel)
    for rel in SOURCE_OPTIONAL:
        if (root/rel).exists():
            regular(root/rel)
            source[rel] = sha(root/rel)
    old = verify_previous_release(root)
    viewer = verify_viewer_output(root,old['n_instances'])
    summary = read_json(root/PROTEIN_SUMMARY)
    if summary.get('n_instances') != 12663 or set(summary.get('marker_names',[])) != MARKERS:
        raise RuntimeError('Workflow09 protein summary counts/panel differ')
    input_files = [MASK,SPATIAL_TABLE,PROTEIN_SUMMARY,MORPH_TABLE]
    input_hashes = {}
    for rel in input_files:
        regular(root/rel)
        input_hashes[rel.as_posix()] = sha(root/rel)
    if input_hashes[MASK.as_posix()] != viewer.get('mask_sha256') or input_hashes[MASK.as_posix()] != old['input_mask_sha256']:
        raise RuntimeError('Mask no longer matches viewer/Workflow09')
    if input_hashes[SPATIAL_TABLE.as_posix()] != viewer.get('spatial_table_sha256'):
        raise RuntimeError('Spatial table no longer matches viewer')
    if input_hashes[PROTEIN_SUMMARY.as_posix()] != viewer.get('workflow09_summary_sha256'):
        raise RuntimeError('Protein summary no longer matches viewer')
    prior_outputs = old.get('workflow09c_outputs',{})
    for key in (SPATIAL_TABLE.as_posix(), PROTEIN_SUMMARY.as_posix()):
        if input_hashes[key] != prior_outputs.get(key):
            raise RuntimeError('Viewer inputs differ from frozen Workflow09 release: '+key)
    if old.get('marker_names') and set(old['marker_names']) != MARKERS:
        raise RuntimeError('Workflow09 frozen marker panel differs')
    hotspot = verify_hotspot_qc(root)
    return {
        'release_type':'exploratory_3d_pseudo_instance_visualization_v1',
        'n_instances':viewer['n_instances'],
        'n_marker_channels':5,
        'viewers':{(VIEWER / name).as_posix(): sha(root/VIEWER/name) for name in VIEWER_OUTPUTS},
        'viewer_manifest_sha256':sha(root/VIEWER/'viewer_sha256.json'),
        'input_hashes':input_hashes,
        'hotspot_qc_hashes':hotspot,
        'source_code_hashes':source,
        'workflow09_release_manifest_sha256':sha(root/PREVIOUS_REL/'release_sha256.json'),
        'workflow09_provenance_sha256':sha(root/PREVIOUS_REL/'provenance_lock.json'),
        'version_note':'Archival v1.0; renderer generated by exploratory viewer v0.1.0',
        'visual_qc_observations':[
            'Rendered centroid and ROI voxel views inspected in conversation; HER2-rich spatial patches visually apparent',
            'Registered HER2/panCK/Ir193 ROI montage visually reviewed at Z=0,5,10,15',
            'Visual checks are qualitative and do not independently establish registration accuracy'
        ],
        'limitations':[
            'A single specimen; watershed pseudo-instances are not validated whole-cell segmentation',
            'XY are pixels and Z is section index, physical Z spacing and 3D aspect ratio unverified',
            'Image-mask registration is visually plausible but not independently validated',
            'Mask-restricted channel values do not establish cellular protein expression/positivity',
            'HER2 top 1% threshold is data-selected for visualization, not validated biological cutoff',
            'Spatial block consistency is descriptive and does not establish independent-sample inference',
            'HER2 ROI montage and block CSV were generated with interactive one-off commands; '
            'canonical regeneration scripts for them have not been archived in this release'
        ]}


def archive_hashes(folder: Path) -> dict:
    return files_hash(folder, 'release_sha256.json')


def make(root: Path,lock: dict) -> Path:
    dest=root/REL
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f'Release already exists; refusing overwrite: {dest}')
    dest.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.workflow10_stage_',dir=dest.parent))
    try:
        for rel in lock['source_code_hashes']:
            original = root/rel
            target = stage/'snapshot'/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(original,target)
        for name in VIEWER_OUTPUTS+['viewer_sha256.json']:
            target=stage/'viewer'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/VIEWER/name,target)
        for name in QC_OUTPUTS:
            target=stage/'hotspot_qc'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/QC/name,target)
        (stage/'provenance_lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (stage/'README_中文.md').write_text(README,encoding='utf-8')
        # Re-audit after copy: abort on any in-flight source changes.
        if preflight(root) != lock:
            raise RuntimeError('Inputs changed during release staging; aborting')
        for rel,digest in lock['source_code_hashes'].items():
            if sha(stage/'snapshot'/rel)!=digest:
                raise RuntimeError('Copied code differs from source: '+rel)
        for name in VIEWER_OUTPUTS:
            if sha(stage/'viewer'/name)!=lock['viewers'][(VIEWER/name).as_posix()]:
                raise RuntimeError('Copied viewer differs from source: '+name)
        if sha(stage/'viewer'/'viewer_sha256.json')!=lock['viewer_manifest_sha256']:
            raise RuntimeError('Copied viewer digest manifest changed')
        for name in QC_OUTPUTS:
            if sha(stage/'hotspot_qc'/name)!=lock['hotspot_qc_hashes'][(QC/name).as_posix()]:
                raise RuntimeError('Copied hotspot artifact differs from source: '+name)
        (stage/'release_sha256.json').write_text(json.dumps(archive_hashes(stage),indent=2)+'\n',encoding='utf-8')
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(f'Release created concurrently: {dest}')
        stage.rename(dest)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return dest


def verify(root: Path) -> dict:
    dest=root/REL
    stored=read_json(dest/'release_sha256.json')
    if not isinstance(stored,dict) or stored!=archive_hashes(dest):
        raise RuntimeError('Frozen Workflow10 release files changed')
    oldlock=read_json(dest/'provenance_lock.json')
    now=preflight(root)
    if oldlock != now:
        raise RuntimeError('Frozen provenance no longer matches live sources/results')
    for rel,digest in oldlock['source_code_hashes'].items():
        if sha(dest/'snapshot'/rel)!=digest:
            raise RuntimeError('Frozen source snapshot changed: '+rel)
    return {'verified_release_files':len(stored),
            'verified_viewer_html':2,
            'verified_hotspot_qc_artifacts':len(QC_OUTPUTS),
            'n_instances':oldlock['n_instances'],
            'n_marker_channels':oldlock['n_marker_channels'],
            'geometry_status':'Z section index ONLY; registration not independently verified'}


README = '''# Workflow 10 — 3D Tissue Microenvironment Visualization v1.0

**状态：已冻结的探索性、离线可视化基线（不代表生物学验证通过）。**

## 本 release 包含

- `viewer/centroid_marker_viewer.html`：12,663 个作者 watershed 伪实例质心、五通道信号与 QC 交互选择。
- `viewer/roi_voxel_viewer.html`：前景最密集 192 × 192 px ROI 的抽样掩码体素交互图。
- `viewer/viewer_summary.json` + `viewer/viewer_sha256.json`：原始输出信息及哈希。
- `hotspot_qc/her2_hotspot_blocks_120px.csv`：120 px 分块的 HER2 Top 1% 描述性统计。
- `hotspot_qc/her2_core_roi_registered_overlay.png`：X=480:720, Y=480:720, Z=0/5/10/15 的 HER2、panCK、Ir193 叠加核查图。
- `snapshot/`：生成 Viewer 的代码与冻结/测试脚本快照。
- `provenance_lock.json` + `release_sha256.json`：输入关联、SHA-256 和版本完整性。

## 可复现性与局限

- Viewer HTML 的生成代码已存档；Viewer 输入文件仍保存在原项目 `runs/`、`data/` 中。
- HER2 ROI 叠加图与热点 CSV **原先通过交互式单次命令生成，未保存为可独立执行的工作流源码**。本次归档了它们的原件及哈希，不能宣称它们已完全代码级复现。
- XY 为像素坐标，Z 仅为 0–15 切片索引；未验证物理 Z spacing。
- 仅一个标本；伪实例不等于真实完整细胞身份；膜相关蛋白可能不在核样 mask 内。
- 高信号形态与同域出现有视觉支持，不代表 HER2 IHC 评分、细胞阳性表型、精准配准或跨标本推断。
- `viewer/` 内 HTML 为包含 Plotly 运行库的离线文件；可直接在浏览器中打开，无须网络。

## 验证

在项目根目录：`python tools/freeze_workflow10.py --verify`。
验证同时检查 Release 本身和其绑定的现有输入、原始 QC 文件及 Workflow09 release。
离线查看可直接打开 `releases/3d_tissue_microenvironment_visualization_v1.0/viewer/centroid_marker_viewer.html`。
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    flags=ap.add_mutually_exclusive_group()
    flags.add_argument('--apply',action='store_true',help='Create immutable Workflow10 release')
    flags.add_argument('--verify',action='store_true',help='Verify frozen files and live provenance')
    ap.add_argument('--project-root',type=Path,default=Path('.'))
    args=ap.parse_args()
    root=args.project_root.resolve()
    if args.verify:
        print('PASS:',json.dumps(verify(root),ensure_ascii=False));return
    lock=preflight(root)
    print('PREFLIGHT PASS:',lock['n_instances'],'instances;',
          len(lock['viewers']),'viewer outputs;',len(lock['hotspot_qc_hashes']),'hotspot QC artifacts')
    if args.apply:
        print('FROZEN:',make(root,lock))
    else:
        print('DRY-RUN ONLY: use --apply to create a new release (will refuse overwrite)')


if __name__=='__main__':
    main()
