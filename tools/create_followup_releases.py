#!/usr/bin/env python3
"""Create frozen, non-destructive research-stage releases from an existing 3D-TissueMap-AI project.

Default = dry run. --apply creates NEW releases only; never edits existing releases,
source scripts, runs or datasets.  Snapshots are archival: run original scripts from project root.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile

SCRIPT_DIR = 'scripts'
# Ordered release set. Existing six releases are intentionally excluded.
STAGES = [
    {
        'name': '3d_z_sampling_benchmark_v0.1',
        'name_zh': 'Z 向采样与留出截面预测',
        'status': 'manuscript_candidate; scientific validation pending',
        'role': 'Main figures: local-Z sampling; independently report mean and median errors',
        'scripts': [
            'analyze_secondher2_z_lag_decay.py',
            'evaluate_secondher2_nearest2_spatialbootstrap.py',
            'evaluate_secondher2_2d_vs_3d_heldout.py',
            'run_secondher2_core_study.py',
            'run_lvilymph_external_validation.py',
        ],
        'runs': [
            'secondher2_z_lag_profile_v0_1',
            'secondher2_nearest2_spatial_robustness_v0_1',
            'secondher2_2d_vs_3d_heldout_v0_1',
            'secondher2_core_study_v0_1',
            'lvilymph_external_sampling_v0_1',
        ],
        'finding': 'Nearest-2 improves held-out middle-Z signal prediction over Nearest-1 in the tested SecondHer2 and LVILymph volumes. Nearest-2 is NOT universally better than all-other planes.',
        'caveat': 'Fixed pre-existing 3D pseudo-instance geometry includes the held-out plane; this is a signal-holdout experiment, NOT recovery of missing 3D cell geometry. Only two volumes, not verified independent patients.',
        'next': 'Check physical voxel spacing, source provenance, and produce unified cross-volume error figures with per-volume/block uncertainty.',
    },
    {
        'name': '3d_spatial_hotspot_sensitivity_v0.1_exploratory',
        'name_zh': '空间热点与失败对照（探索性）',
        'status': 'exploratory; negative/geometry-dependent results; supplemental only',
        'role': 'Supplementary results: geometry-sensitive proxy hotspot reference and no successful combined-risk filtering',
        'scripts': [
            'qc_secondher2_3d_spatial_discordance.py',
            'qc_secondher2_spatial_robustness.py',
            'run_secondher2_hotspot_recovery.py',
            'run_secondher2_hotspot_transfer.py',
        ],
        'runs': [
            'secondher2_3d_spatial_discordance_v0_1',
            'secondher2_3d_spatial_robustness_v0_1',
            'secondher2_3d_hotspot_recovery_v0_1',
            'secondher2_reliability_hotspot_transfer_v0_1',
        ],
        'finding': 'Hotspot prediction exceeds shuffle null, but XYZ-reference advantage reverses with XY-reference. Combined reliability filter underperforms hotspot margin.',
        'caveat': 'Reference hotspot labels are heldout-plane PROXIES using spatial neighbourhood definitions, not independent biological ground truth. Never claim 3D > 2D from reference-matched geometry.',
        'next': 'Keep as transparent exploratory/supplemental evidence; do not retune on this sample.',
    },
    {
        'name': '3d_measurement_reliability_v0.1',
        'name_zh': '测量可靠性与风险预测',
        'status': 'manuscript_candidate; portable model is signal-driven; ablation established',
        'role': 'Main/Supplementary: reliability scores and source spatial OOF; no claim of independent geometry benefit',
        'scripts': [
            'run_secondher2_reliability.py',
            'run_portable_reliability_transfer.py',
            'run_portable_reliability_ablation.py',
        ],
        'runs': [
            'secondher2_reliability_model_v0_1',
            'lvilymph_portable_reliability_transfer_v0_1',
            'lvilymph_portable_reliability_ablation_v0_1',
        ],
        'finding': 'Frozen signal-only model transfers reasonably for some markers; full vs signal-only has no clear primary improvement. Removing absolute intensity materially changes performance.',
        'caveat': '10 portable features; source-only fitting. Source/target absolute intensity distributions strongly shift, and LYVE-1 fails. Risk is heldout intensity error, not validated biological measurement truth.',
        'next': 'Report source-trained baseline, signal-only and full ablations with spatial uncertainty; freeze model and disclose OOD dependence.',
    },
    {
        'name': '3d_cross_volume_validation_v0.1',
        'name_zh': '跨体积验证（LVILymph）',
        'status': 'completed computational tests; independent biological replication unverified',
        'role': 'Main results: independent imaging-volume replication and marker-specific risk transfer',
        'scripts': [
            'run_lvilymph_external_validation.py',
            'run_portable_reliability_transfer.py',
            'run_portable_reliability_ablation.py',
        ],
        'runs': [
            'lvilymph_external_sampling_v0_1',
            'lvilymph_portable_reliability_transfer_v0_1',
            'lvilymph_portable_reliability_ablation_v0_1',
        ],
        'finding': 'LVILymph 16 planes, 29 aligned channels; 1,780 eligible pseudo-instances; 5 tested marker channels show nearest2 > nearest1. Main target Gd156 external full AUROC 0.7884; signal-only 0.7846.',
        'caveat': 'Source labels Eu151 != LVILymph Eu151 (CD31). Target Gd156 c-erbB- identity provisional. Missing original 17th aligned plane, pixel size and independent patient origin require verification.',
        'next': 'Verify panel provenance, patient/case independence, physical Z spacing, masks/registration and out-of-distribution scale effects.',
    },
    {
        'name': '3d_manuscript_evidence_v0.1',
        'name_zh': '论文图表与可复现证据归档',
        'status': 'evidence collection only; manuscript NOT submitted or final',
        'role': 'Evidence snapshots and stage-specific result inventory for Figures 1-5, Methods and Supplementary',
        'scripts': [],
        'runs': [
            'workflow12_secondher2_fixed_boundary_transfer_all92_v0_1',
            'secondher2_3d_object_protein_quantification_v0_1',
            'secondher2_3d_spatial_robustness_v0_1',
            'secondher2_core_study_v0_1',
            'secondher2_3d_hotspot_recovery_v0_1',
            'secondher2_reliability_model_v0_1',
            'secondher2_reliability_hotspot_transfer_v0_1',
            'lvilymph_external_sampling_v0_1',
            'lvilymph_portable_reliability_transfer_v0_1',
            'lvilymph_portable_reliability_ablation_v0_1',
        ],
        'finding': 'Candidate Figure 1 data and instance provenance; Figure 2 spatial structure; Figure 3 Z-sampling; Figure 4 measurement risk; Figure 5 cross-volume transfer and ablations.',
        'caveat': 'Candidate figures not publication-ready by default. Must check axis labels, measured-vs-pseudo-instance terminology, paper-level unit of replication, physical spacing and permissions.',
        'next': 'Select final result definitions, regenerate figures from frozen tables, prepare Methods, Results, data/code availability and target-journal formatting.',
    },
]

ALLOWED_RESULTS = {'.csv', '.json', '.md', '.txt', '.tsv', '.png', '.pdf', '.svg', '.yaml', '.yml'}
DEFAULT_MAX_FILE_MB = 12
DEFAULT_MAX_STAGE_MB = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            digest.update(part)
    return digest.hexdigest()


def safe_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def pick_scripts(project: Path, stage: dict):
    found = []
    missing = []
    for basename in stage['scripts']:
        candidates = [project / 'scripts' / basename, project / basename]
        match = next((p for p in candidates if p.is_file() and not p.is_symlink()), None)
        if match is None:
            missing.append(basename)
        else:
            found.append(match)
    return found, missing


def stage_input_candidates(project: Path, stage: dict, limit_file: int, limit_stage: int):
    files, skipped, missing_runs = [], [], []
    total = 0
    for run_name in stage['runs']:
        run = project / 'runs' / run_name
        if not run.is_dir():
            missing_runs.append(run_name)
            continue
        for p in sorted(run.rglob('*')):
            if not p.is_file() or p.is_symlink():
                continue
            if p.suffix.lower() not in ALLOWED_RESULTS:
                continue
            size = p.stat().st_size
            rel = safe_rel(p, project)
            if size > limit_file:
                skipped.append((rel, 'file size cap', size))
            elif total + size > limit_stage:
                skipped.append((rel, 'release size cap', size))
            else:
                files.append(p)
                total += size
    return files, skipped, missing_runs


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def build_stage(project: Path, target: Path, stage: dict, limit_file: int, limit_stage: int, apply: bool):
    scripts, missing_scripts = pick_scripts(project, stage)
    results, skipped, missing_runs = stage_input_candidates(project, stage, limit_file, limit_stage)
    print(f"\n[{stage['name']}] scripts {len(scripts)}/{len(stage['scripts'])}, "
          f"runs {len(stage['runs'])-len(missing_runs)}/{len(stage['runs'])}, "
          f"evidence files {len(results)}")
    if missing_scripts:
        print('  Missing script snapshots:', ', '.join(missing_scripts))
    if missing_runs:
        print('  Missing run folders:', ', '.join(missing_runs))
    if not apply:
        return
    if target.exists():
        raise FileExistsError(f'Release already exists; no overwrite: {target}')
    temp = Path(tempfile.mkdtemp(prefix='.new_release_', dir=str(target.parent)))
    try:
        manifest = {
            'schema': '3d_tissuemap_followup_release_v0_1',
            'release': stage['name'],
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'project_root_at_snapshot': str(project.resolve()),
            'status': stage['status'],
            'stage': stage['name_zh'],
            'purpose': stage['role'],
            'missing_scripts': missing_scripts,
            'missing_runs': missing_runs,
            'skipped_evidence': [{'path': a, 'reason': b, 'bytes': c} for a, b, c in skipped],
            'file_sha256': [],
            'original_run_dirs': ['runs/' + r for r in stage['runs']],
            'snapshot_warning': 'Copied files are archived evidence only. Scripts may import peers from original project/scripts and must run from project root.',
        }
        copied = []
        for source in scripts + results:
            relative = safe_rel(source, project)
            dest = (temp / 'snapshots' / relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            item = {'original_project_path': relative,
                    'snapshot_path': ('snapshots/' + relative),
                    'size_bytes': source.stat().st_size, 'sha256': sha256(dest)}
            copied.append(item)
        manifest['file_sha256'] = copied
        readme = f"# {stage['name']}\n\n**科研阶段：** {stage['name_zh']}\n\n**状态：** {stage['status']}\n\n"
        readme += f"**定位：** {stage['role']}\n\n## 冻结的实验结论（需要用原始结果核对）\n\n{stage['finding']}\n\n"
        readme += f"## 关键限制\n\n{stage['caveat']}\n\n## 后续工作\n\n{stage['next']}\n\n"
        readme += '## 原始实验结果目录\n\n' + ''.join(f'- `runs/{r}/`' + (' **[未找到]**' if r in missing_runs else '') + '\n' for r in stage['runs']) + '\n'
        readme += '## 代码快照\n\n' + (''.join(f'- `snapshots/{safe_rel(p,project)}`\n' for p in scripts) or '无需新增代码快照。\n')
        if missing_scripts:
            readme += '\n**未找到脚本：** ' + ', '.join(f'`{s}`' for s in missing_scripts) + '\n'
        readme += '\n## 可复现性要求\n\n'
        readme += ('1. 原项目根目录的 `scripts/`、`runs/`、`data/` 均未移动或修改。\n'
                   '2. 快照脚本只用于冻结版本；如果脚本动态载入同级模块，必须在原项目根目录运行。\n'
                   '3. `manifest.json` 列出复制文件的 SHA256；大型原始 TIFF 和掩膜没有复制。\n'
                   '4. 数千个伪实例不构成数千个生物学重复，论文统计单位和外部样本独立性需要说明。\n')
        write_text(temp/'README.md', readme)
        write_text(temp/'manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        # rename after all files were copied, preventing partial final releases
        temp.rename(target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def roadmap(project: Path) -> str:
    lines = [
        '# 3D-TissueMap AI：正式科研发布目录与后续工作路线',
        '',
        '> 这是 `releases/` 中的科研阶段归档索引，不改变现有脚本运行路径，也不代表全部任务已完成。',
        '',
        '## 既有发布（完全保留）',
        '',
        '| 现有目录 | 研究阶段 |', '|---|---|',
    ]
    existing = [
        ('3d_instance_segmentation_v1.0', '三维实例分割'),
        ('3d_cell_morphometry_v1.0', '细胞三维形态测量'),
        ('3d_protein_marker_integration_v1.0', '蛋白标记物整合'),
        ('3d_spatial_organization_v1.0', '空间结构组织分析'),
        ('3d_spatial_protein_attribution_v0.2_exploratory', '空间蛋白归属探索'),
        ('3d_tissue_microenvironment_visualization_v1.0', '组织微环境可视化'),
    ]
    for name, desc in existing:
        present = (project/'releases'/name).is_dir()
        lines.append(f'| `{name}` | {desc}' + ('' if present else '（当前未检测到）') + ' |')
    lines += ['', '## 新增研究阶段', '', '| 发布目录 | 核心任务 | 状态 |', '|---|---|---|']
    for x in STAGES:
        lines.append(f"| `{x['name']}` | {x['name_zh']} | {x['status']} |")
    lines += [
        '', '## 尚未解决的投稿前检查（必须人工核实）', '',
        '1. 确认 SecondHer2 和 LVILymph 是否来自独立患者/独立采集，避免把两个体积或 1,780 个对象称作独立临床队列。',
        '2. 核实 LVILymph 的物理 Z 间距、像素尺寸、通道来源以及第 17 张原始 OME 未进入 16 层配准体的原因。',
        '3. 核实 LVILymph `Gd156` 原始截断标签 `c-erbB-` 的完整抗体名称；Eu151 在 LVILymph 是 CD31，不能套用 SecondHer2 通道表。',
        '4. 论文中区分留出层强度代理误差与独立生物学真值；跨层伪实例来自完整分割体积。',
        '5. 报告 Full vs Signal-only 的无明确几何增益、去除绝对强度的性能损失、LYVE-1 转移失败和 hotspot filtering 失败。',
        '6. 采用 `releases/3d_manuscript_evidence_v0.1/` 内的证据快照组织候选图；输出正式 PDF/SVG，检查显微标尺、通道名称和统计单位。',
        '', '## 原项目代码入口', '',
        '运行正式实验仍应在项目根目录使用 `python scripts/<script>.py`，不要在 `releases/` 快照子目录直接运行。',
        '', '## 防误操作与更新规则', '',
        '本工具只创建新的发布目录和本索引，不覆盖或移动任何旧发布、`scripts/`、`runs/` 或原始数据。'
        ' 若未来需要新版本，请复制工具并修改版本后新建目录；不要静默修改已冻结的旧版本。',
        '',
    ]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path('.'), help='3D-TissueMap-AI project root')
    p.add_argument('--apply', action='store_true', help='Create new releases; dry-run without this flag')
    p.add_argument('--max-file-mb', type=int, default=DEFAULT_MAX_FILE_MB)
    p.add_argument('--max-stage-mb', type=int, default=DEFAULT_MAX_STAGE_MB)
    args = p.parse_args()
    project = args.project_root.resolve()
    if not (project/'scripts').is_dir() or not (project/'runs').is_dir():
        p.error(f'Not a recognizable project root (missing scripts/ or runs/): {project}')
    target_root = project/'releases'
    if not target_root.is_dir():
        p.error(f'Expected existing releases/ directory, not found: {target_root}')
    if args.max_file_mb < 1 or args.max_stage_mb < 1:
        p.error('Positive MB limits required')
    max_file = args.max_file_mb * 1024 * 1024
    max_stage = args.max_stage_mb * 1024 * 1024
    print(f'PROJECT: {project}\nMODE: {"APPLY / COPY snapshots" if args.apply else "DRY-RUN"}')
    if args.apply:
        planned = [target_root/x['name'] for x in STAGES] + [target_root/'RELEASE_ROADMAP_v0.1.md']
        exists = [str(p) for p in planned if p.exists()]
        if exists:
            raise SystemExit('Refusing to modify an existing release. Existing paths:\n  ' + '\n  '.join(exists))
    for stage in STAGES:
        build_stage(project, target_root/stage['name'], stage, max_file, max_stage, args.apply)
    if args.apply:
        write_text(target_root/'RELEASE_ROADMAP_v0.1.md', roadmap(project))
        print('\nSUCCESS: created 5 new versioned releases + releases/RELEASE_ROADMAP_v0.1.md')
    else:
        print('\nDRY-RUN ONLY. To create new release directories, add --apply.')
        print('No existing release or script/run/data was changed.')

if __name__ == '__main__':
    main()
