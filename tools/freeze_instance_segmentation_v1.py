#!/usr/bin/env python3
"""Non-destructive, provenance-aware freeze for 3D-TissueMap instance segmentation v1.0.

Dry run by default. Freeze creates only releases/3d_instance_segmentation_v1.0.
Parent ZIP archival is separately opt-in and does not affect the project source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

VERSION = '3d_instance_segmentation_v1.0'
KNOWN_RUNS = [
    'segmentation_extended', 'voxel_affinity_extended', 'affinity_fixed_patch_900',
    'affinity_graph_pilot', 'affinity_graph_diagnostics', 'affinity_region_graph_pilot',
    'region_graph_1x4x4', 'region_graph_1x3x3', 'supervoxel_purity_audit',
    'supervoxel_oracle_merging', 'region_candidate_audit', 'dynamic_region_graph_train',
    'region_oracle_connectivity', 'foreground_graph_attribution',
    'region_interface_dataset_pilot', 'region_interface_dataset_train',
    'region_merge_predictor_baselines', 'learned_region_graph_train_smoke',
    'learned_region_graph_val_diagnostic', 'instance_error_attribution_val',
    'z_continuity_rescue_train_smoke', 'z_continuity_rescue_val_diagnostic',
    'z_link_safety_audit', 'z_registration_identity_audit',
    'cross_slice_identity_review_v2',
]
RUN_TEXT_EXT = {'.json', '.csv', '.tsv', '.md', '.txt', '.yaml', '.yml', '.toml', '.log'}
RUN_MODEL_EXT = {'.pt', '.pth', '.ckpt', '.joblib', '.pkl', '.pickle', '.safetensors'}
RUN_FIGURE_EXT = {'.png', '.jpg', '.jpeg', '.svg', '.pdf'}
CODE_EXT = {'.py', '.sh', '.md', '.txt', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
ROOT_CONFIG = {'README.md', 'README_中文.md', 'pyproject.toml', 'requirements.txt',
               'requirements-dev.txt', 'environment.yml', 'environment.yaml',
               'pytest.ini', 'setup.cfg', 'START_HERE_中文.md'}
FIELDS = ['workflow', 'file', 'bytes', 'sha256']


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def workflow_for(filename: str) -> str:
    s = filename.lower().replace('-', '_')
    if any(x in s for x in ('cross_slice', 'z_link', 'z_continuity', 'registration_identity', 'z_registration')):
        return '06_z_identity_and_linking'
    if any(x in s for x in ('spatial_morph', 'morphometry', 'protein_marker', 'spatial_organization')):
        return '90_next_workflow_not_v1_core'
    if any(x in s for x in ('instance_error', 'attribution', 'foreground_graph', 'audit_foreground')):
        return '05_failure_attribution'
    if any(x in s for x in ('region', 'supervoxel', 'graph', 'oracle_merging')):
        return '04_region_graph_and_merge_learning'
    if 'affinity' in s:
        return '03_voxel_affinity'
    if any(x in s for x in ('instance', 'watershed', 'center_boundary', 'head_', 'nucleus')):
        return '02_instance_baselines'
    if any(x in s for x in ('segmentation', 'unet', 'semantic')):
        return '01_semantic_segmentation'
    if any(x in s for x in ('qc', 'data', 'stack', 'preprocess', 'audit', 'tile')):
        return '00_data_qc'
    return '99_shared_or_unclassified'


def stable_files(folder: Path, exts: set[str]):
    if not folder.is_dir():
        return
    for p in sorted(folder.rglob('*')):
        if (p.is_file() and not p.is_symlink() and p.suffix.lower() in exts
                and not any(part in {'__pycache__', '.git', '.pytest_cache', '.ipynb_checkpoints'}
                            for part in p.parts)):
            yield p


def selected_sources(project: Path, *, max_file_mb: int, include_figures: bool,
                     include_models: bool):
    """Yield (source path, destination relative to release, workflow); no source mutations."""
    max_bytes = max_file_mb * 1024 * 1024
    for folder in ('scripts', 'tests', 'tools'):
        root = project / folder
        for f in stable_files(root, CODE_EXT):
            rel = f.relative_to(project)
            wf = workflow_for(f.name) if folder == 'scripts' else f'99_{folder}'
            yield (f, Path('snapshot') / rel, wf, max_bytes)
    for name in ROOT_CONFIG:
        f = project / name
        if f.is_file() and not f.is_symlink():
            yield (f, Path('snapshot') / 'project_configs' / f.name,
                   '99_project_config', max_bytes)
    for name in KNOWN_RUNS:
        root = project / 'runs' / name
        if not root.is_dir():
            continue
        suffixes = RUN_TEXT_EXT.copy()
        if include_figures:
            suffixes |= RUN_FIGURE_EXT
        if include_models:
            suffixes |= RUN_MODEL_EXT
        for f in stable_files(root, suffixes):
            # Never add large raw volumes, and do not follow symlinks outside the repo.
            rel = f.relative_to(project)
            yield (f, Path('snapshot') / rel, 'experiment_results', max_bytes)


def write_csv(path: Path, headers: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)



def extract_metric_rows(project: Path) -> list[dict]:
    """Copy reported values from local CSVs, without running inference or selecting a winner."""
    specs = [
        ('learned_region_graph_val_diagnostic/learned_region_instance_comparison.csv',
         'region_graph_val_development'),
        ('z_continuity_rescue_val_diagnostic/z_link_rescue_comparison.csv',
         'z_rescue_val_development'),
    ]
    results = []
    for subpath, stage in specs:
        src = project / 'runs' / subpath
        if not src.is_file():
            continue
        with src.open('r', encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                results.append({
                    'source_file': f'runs/{subpath}', 'stage': stage,
                    'split': row.get('split',''), 'tile_y': row.get('tile_y',''),
                    'tile_x': row.get('tile_x',''), 'method': row.get('method',''),
                    'pq': row.get('pq',''), 'tp': row.get('tp',''),
                    'fp': row.get('fp',''), 'fn': row.get('fn',''),
                    'n_reference': row.get('n_reference',''),
                    'n_predicted': row.get('n_predicted',''),
                    'interpretation': 'Single specimen, repeatedly reused validation and author watershed pseudo IDs',
                })
    return results


def workflow_map_markdown(scripts: list[dict]) -> str:
    grouped = {}
    for item in scripts:
        grouped.setdefault(item['workflow'], []).append(item['script'])
    lines = [
        '# 脚本工作流索引（冻结时扫描生成）', '',
        '这里是**索引**，不是重命名脚本或搬迁可执行路径。代码保存在 `snapshot/scripts/`。',
        '原项目中的 `scripts/` 位置未更改，避免破坏已有导入路径和历史命令。', '',
        '## 推荐复现顺序', '',
        '数据 QC → 语义分割 → 实例基线 → Voxel Affinity → Region Graph 与学习合并 → 错误归因 → Z 向身份审计。',
        '本项目实验属于单标本探索，结果不能表述为独立临床验证。', '',
    ]
    for wf in sorted(grouped):
        lines.append(f'## {wf}')
        for name in sorted(grouped[wf]):
            lines.append(f'- [`{name}`](snapshot/scripts/{name})')
        lines.append('')
    return '\n'.join(lines)


def git_info(project: Path) -> dict:
    if not (project / '.git').exists():
        return {'git_repo': False}
    result = {'git_repo': True}
    for field, args in [('git_commit', ['rev-parse','HEAD']),
                        ('git_worktree_status', ['status','--porcelain'])]:
        try:
            p = subprocess.run(['git', '-C', str(project), *args], capture_output=True,
                               text=True, timeout=10, check=True)
            result[field] = p.stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            result[field] = None
    return result

def build_readme(stats: dict) -> str:
    return f'''# 3D Instance Segmentation v1.0｜正式冻结说明

冻结版本：`{VERSION}`；生成时间（UTC）：{stats['generated_utc']}。

## 研究状态（必须保留）

- **已完成的工程阶段**：三维语义分割、Watershed、Voxel Affinity、Supervoxel、Static/Dynamic/Learned RAG、Z-Continuity Rescue、误差归因及 Z 配准/身份审计。
- **不是已验证的最优三维实例分割算法**：用户报告同一反复使用的 VAL 图块 `(384,576)` 上，Distance Watershed PQ ≈ **0.118384**；Learned Logistic RAG PQ ≈ **0.083777**；Learned RAG + Z Geometry Rescue PQ ≈ **0.099232**。Geometry Rescue 匹配实例 TP 从 404 降至 294，不能只看 PQ 增加。
- 参考均为**作者生成的 Watershed pseudo-instance labels**，而非独立人工 3D cell ground truth。
- 仅**单份 3D IMC 标本**，TRAIN 空间图块有关联；VAL 多次用于开发，不是干净的独立测试；**TEST 不得用本冻结工具访问、重评或调参**。
- Ir193 图像/掩码跨切片精确配准、物理 Z-spacing、实例 ID 与 protein CSV 的对应关系仍未完成独立核实。

## 冻结内容

- `snapshot/scripts/` 与 `snapshot/tests/`：保留原项目路径结构的代码快照；如果原项目包含后续 Morphology 脚本，其归类为“下一工作流”，不代表纳入 v1.0 的方法学结论。
- `snapshot/runs/`：仅拷贝选定实验目录中的文本结果/CSV/JSON、（按本次运行选项）图像与 checkpoint；**不会拷贝原始数据或大型中间体数据**。
- `script_workflow_index.csv` 和 `WORKFLOW_MAP_中文.md`：按科研工作流对源脚本分类。源 `scripts/` 未搬动，以免打断 Python 导入。
- `frozen_metric_comparison.csv`：直接读取本地 VAL CSV 汇总已有 PQ，不重算、不选择最优阈值。
- `MANIFEST_SHA256.csv`：每个冻结文件的 SHA-256；用于审计来源及验证未被修改。
- `excluded_files.csv`：因体积上限而未快照的文件。缺失关键模型需解决后才能声称完整恢复环境。
- `source_run_inventory.csv`：已有实验目录与文件统计（仅项目本地）。
- `runtime_info.json` 和 `pip_freeze.txt`（如成功）：记录冻结当时的 Python 环境；不等于完整 CUDA/Conda 环境复现。

快照包含文件：**{stats['files_copied']}**；总计 **{stats['bytes_copied']/1024**2:.1f} MiB**；因文件大小限制跳过 **{stats['files_excluded']}** 个文件。

## 用法与安全

1. **继续在项目根目录运行原脚本**，不要把原始 `scripts/`、`tests/`、`runs/` 直接移动到工作流分类目录。冻结快照用于回溯、审计和存档。
2. 原始 IMC 数据仍在 `data/`，没有复制到 release。真正跨机器重跑还需要数据来源、hash、路径映射以及受检模型的环境依赖。
3. 用户应手动执行本项目关键测试，并核查 `excluded_files.csv` 和 manifest；本工具**不执行任何模型训练或独立重复结果验证**。
4. 如需校验冻结后的快照：`python tools/freeze_instance_segmentation_v1.py --verify-release`。
5. ZIP 增量包只在**单独显式**运行 `--archive-zips-only --apply` 时移入项目父目录的 `_archive/instance_segmentation_v1_updates/`，不会自动删除。

**下一独立工作流**：`3D Cell Morphometry`。继续使用 `runs/` 新路径，避免修改此冻结 release。
'''


def summarize_run_inventory(project: Path):
    rows = []
    runs = project / 'runs'
    if not runs.is_dir():
        return rows
    for subdir in sorted(runs.iterdir()):
        if not subdir.is_dir() or subdir.is_symlink():
            continue
        files = sum(1 for f in subdir.rglob('*') if f.is_file() and not f.is_symlink())
        rows.append({'run_dir': str(subdir.relative_to(project)),
                     'n_files': files, 'selected_for_snapshot': subdir.name in KNOWN_RUNS})
    return rows


def freeze(project: Path, *, apply: bool, max_file_mb: int, include_figures: bool,
           include_models: bool):
    if not (project / 'scripts').is_dir():
        raise SystemExit(f'ERROR: project scripts/ not found: {project}. Run from repository root.')
    release = project / 'releases' / VERSION
    if release.exists():
        raise SystemExit(f'ERROR: freeze already exists; refusing to overwrite: {release}')
    sources = list(selected_sources(project, max_file_mb=max_file_mb,
                                    include_figures=include_figures,
                                    include_models=include_models))
    included = [it for it in sources if it[0].stat().st_size <= it[3]]
    excluded = [it for it in sources if it[0].stat().st_size > it[3]]
    print(f'Project: {project}\nRelease: {release}')
    print(f'Freeze candidates: {len(included)} files; skipped oversized: {len(excluded)}')
    print(f'Options: figures={include_figures}, models={include_models}, max_file_mb={max_file_mb}')
    print(f'Run directories found: {sum((project / "runs" / x).is_dir() for x in KNOWN_RUNS)}/{len(KNOWN_RUNS)} known')
    if not apply:
        print('DRY-RUN ONLY. Add --apply to create snapshot. Nothing changed.')
        return
    release.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{VERSION}.staging_', dir=release.parent))
    try:
        manifest = []
        scripts = []
        for src, rel, wf, limit in included:
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            sha = checksum(src)
            if sha != checksum(dst):
                raise IOError(f'Hash mismatch in frozen copy: {src}')
            entry = {'workflow': wf, 'file': rel.as_posix(),
                     'bytes': src.stat().st_size, 'sha256': sha}
            manifest.append(entry)
            if len(rel.parts) > 1 and rel.parts[1] == 'scripts':
                scripts.append({'workflow': wf,
                                'script': rel.relative_to(Path('snapshot/scripts')).as_posix(),
                                'sha256': sha})
        write_csv(stage / 'script_workflow_index.csv',
                  ['workflow', 'script', 'sha256'], sorted(scripts, key=lambda r: (r['workflow'], r['script'])))
        (stage / 'WORKFLOW_MAP_中文.md').write_text(workflow_map_markdown(scripts), encoding='utf-8')
        write_csv(stage / 'frozen_metric_comparison.csv',
                  ['source_file','stage','split','tile_y','tile_x','method','pq','tp','fp','fn',
                   'n_reference','n_predicted','interpretation'], extract_metric_rows(project))
        write_csv(stage / 'excluded_files.csv', ['source_file', 'size_bytes', 'reason'], [
            {'source_file': it[0].relative_to(project).as_posix(),
             'size_bytes': it[0].stat().st_size,
             'reason': f'File exceeds --max-file-mb={max_file_mb}'} for it in excluded])
        write_csv(stage / 'source_run_inventory.csv',
                  ['run_dir', 'n_files', 'selected_for_snapshot'], summarize_run_inventory(project))
        generated = datetime.now(timezone.utc).isoformat()
        info = {'generated_utc': generated, 'files_copied': len(manifest),
                'bytes_copied': sum(i['bytes'] for i in manifest),
                'files_excluded': len(excluded),
                'project_path_at_freeze': str(project),
                'python_executable': sys.executable, 'python_version': sys.version,
                'platform': platform.platform(),
                'conda_prefix': os.environ.get('CONDA_PREFIX'),
                'models_included': include_models, 'figures_included': include_figures,
                'user_reports_not_independently_recomputed': True, 'git': git_info(project)}
        (stage / 'runtime_info.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
        try:
            pip = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                                 text=True, timeout=25, check=True)
            (stage / 'pip_freeze.txt').write_text(pip.stdout, encoding='utf-8')
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as exc:
            (stage / 'pip_freeze_unavailable.txt').write_text(str(exc), encoding='utf-8')
        (stage / 'README_冻结说明.md').write_text(build_readme(info), encoding='utf-8')
        # Include generated release metadata in the manifest, not only copied source files.
        source_workflows = {r['file']: r['workflow'] for r in manifest}
        release_manifest = []
        for f in sorted(stage.rglob('*')):
            if f.is_file() and f.name != 'MANIFEST_SHA256.csv':
                key = f.relative_to(stage).as_posix()
                release_manifest.append({'workflow': source_workflows.get(key, 'release_metadata'),
                                         'file': key, 'bytes': f.stat().st_size,
                                         'sha256': checksum(f)})
        write_csv(stage / 'MANIFEST_SHA256.csv', FIELDS, release_manifest)
        # Safe to rename after *all* files are copied and checksums validated.
        stage.rename(release)
        print(f'FROZEN: {release}\nTo verify: python tools/freeze_instance_segmentation_v1.py --verify-release')
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_release(project: Path):
    release = project / 'releases' / VERSION
    manifest = release / 'MANIFEST_SHA256.csv'
    if not manifest.is_file():
        raise SystemExit('No freeze manifest found. Run --apply first.')
    checked = 0
    with manifest.open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            rel = Path(row['file'])
            if rel.is_absolute() or '..' in rel.parts:
                raise RuntimeError(f'Unsafe manifest entry: {rel}')
            path = release / rel
            if not path.is_file() or checksum(path) != row['sha256']:
                raise RuntimeError(f'RELEASE INTEGRITY FAILURE: {rel}')
            checked += 1
    print(f'PASS: {checked} files match SHA-256 manifest: {release}')


def archive_parent_zips(project: Path, *, apply: bool):
    parent = project.parent
    archive = parent / '_archive' / 'instance_segmentation_v1_updates'
    zips = sorted(p for p in parent.glob('3D-TissueMap*.zip') if p.is_file() and not p.is_symlink())
    print(f'Found {len(zips)} parent ZIP packages. Destination: {archive}')
    if not apply:
        print('DRY-RUN ONLY; add --apply --archive-zips-only after freeze verification.')
        return
    if not (project / 'releases' / VERSION / 'MANIFEST_SHA256.csv').exists():
        raise SystemExit('Cannot archive ZIPs before verified freeze exists.')
    verify_release(project)
    archive.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in zips:
        dst = archive / p.name
        if dst.exists():
            # Fail safe: if a package of same name exists, do not overwrite or remove either.
            print(f'SKIP: destination already exists: {dst}')
            continue
        source_hash = checksum(p)
        shutil.copy2(p, dst)
        if checksum(dst) != source_hash:
            dst.unlink(missing_ok=True)
            raise IOError(f'Archive hash mismatch: {p}')
        p.unlink()  # only after checksum-verified archival copy
        moved += 1
    print(f'ARCHIVED: {moved} ZIPs; kept other parent files untouched.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path.cwd(), help='Existing repository root')
    p.add_argument('--apply', action='store_true', help='Perform changes; otherwise dry run')
    p.add_argument('--verify-release', action='store_true', help='Verify SHA256 freeze integrity')
    p.add_argument('--archive-zips-only', action='store_true', help='Separately archive parent update ZIPs after release exists')
    p.add_argument('--max-file-mb', type=int, default=1024, help='Safety limit per copied file (default 1024 MiB)')
    p.add_argument('--no-figures', action='store_true', help='Do not snapshot selected run figures')
    p.add_argument('--no-models', action='store_true', help='Do not snapshot model checkpoints (partial freeze)')
    args = p.parse_args(argv)
    if args.max_file_mb < 1:
        p.error('--max-file-mb must be >=1')
    project = args.project.expanduser().resolve()
    if args.verify_release:
        verify_release(project)
    elif args.archive_zips_only:
        archive_parent_zips(project, apply=args.apply)
    else:
        freeze(project, apply=args.apply, max_file_mb=args.max_file_mb,
               include_figures=not args.no_figures, include_models=not args.no_models)

if __name__ == '__main__':
    main()
