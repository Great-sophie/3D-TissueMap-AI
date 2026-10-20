#!/usr/bin/env python3
"""Read-only Workflow 11 research prototype release organizer.

It never alters scripts/, tests/, configs/, runs/, or previous release directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

RELEASE_NAME = '3d_spatial_protein_attribution_v0.2_exploratory'
RUNS = [
    '3d_spatial_signal_attribution_v0_1',
    '3d_signal_attribution_sensitivity_v0_1',
    '3d_signal_attribution_shift_controls_v0_2',
    '3d_signal_attribution_boundary_robustness_v0_3',
    '3d_membrane_localization_feasibility_v0_1',
    '3d_cadherin_boundary_feasibility_v0_1',
    '3d_cadherin_assisted_celllike_segmentation_v0_1',
    '3d_nucleus_constrained_celllike_segmentation_v0_2',
    '3d_boundary_validation_diagnostics_v0_1',
]
OPTIONAL_RUNS = ['3d_her2_hotspot_qc_v1']
REQUIRED_SCRIPTS = [
    'run_workflow11_boundary_robustness.py',
    'run_workflow11e_membrane_feasibility.py',
    'run_workflow11g_cadherin_segmentation.py',
    'run_workflow11h_nucleus_constrained_segmentation.py',
    'run_workflow11_shift_controls.py',
]
REQUIRED_TESTS = [
    'test_workflow11_attribution.py',
    'test_workflow11_boundary_robustness.py',
    'test_workflow11e_membrane_feasibility.py',
    'test_workflow11g_cadherin_segmentation.py',
    'test_workflow11h_nucleus_constrained_segmentation.py',
    'test_workflow11_shift_controls.py',
]
CURATED_SUFFIXES = {'.csv','.tsv','.json','.md','.txt','.png','.jpg','.jpeg','.svg','.pdf','.html'}
MAX_SINGLE_COPY = 8 * 1024 * 1024
MAX_TOTAL_COPIES = 80 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_regular(path: Path, root: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    return path.resolve().is_relative_to(root.resolve())


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def files_inside(folder: Path, root: Path):
    return sorted(p for p in folder.rglob('*') if safe_regular(p, root))


def detect_source_files(root: Path):
    sources = []
    for name in REQUIRED_SCRIPTS:
        sources.append(root / 'scripts' / name)
    for name in REQUIRED_TESTS:
        sources.append(root / 'tests' / name)
    # Workflow 11A producer may have a different name; discover it from its output names.
    sd = root / 'scripts'
    if sd.is_dir():
        for p in sorted(sd.glob('*.py')):
            if p.name == 'finalize_workflow11_release.py' or p in sources or p.is_symlink():
                continue
            try:
                txt = p.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            if ('3d_spatial_signal_attribution_v0_1' in txt or
                    'instance_marker_region_signal.csv' in txt):
                sources.append(p)
    config_dir = root / 'configs'
    if config_dir.is_dir():
        for p in sorted(config_dir.iterdir()):
            if not safe_regular(p, root):
                continue
            lower = p.name.lower()
            if any(s in lower for s in ('workflow11', 'cadherin', 'protein_marker_manifest', 'spatial_signal')):
                sources.append(p)
    for name in ('pyproject.toml', 'requirements.txt', 'environment.yml', 'pytest.ini'):
        path = root / name
        if safe_regular(path, root):
            sources.append(path)
    # De-duplicate without relying on alphabetical filenames being consistent.
    return sorted(set(sources))


def required_paths(root: Path):
    return ([root/'scripts'/n for n in REQUIRED_SCRIPTS] +
            [root/'tests'/n for n in REQUIRED_TESTS] +
            [root/'runs'/n for n in RUNS])


def preliminary_check(root: Path) -> dict:
    source_files = detect_source_files(root)
    attribution_candidates = [relpath(p, root) for p in source_files
                              if p.parent == (root/'scripts') and p.name not in REQUIRED_SCRIPTS]
    missing = [relpath(p, root) for p in required_paths(root) if not p.exists()]
    bad_symlinks = [relpath(p, root) for p in required_paths(root) if p.is_symlink()]
    return {
        'project_root': str(root),
        'release_target': str(root/'releases'/RELEASE_NAME),
        'required_scripts_found': sum((root/'scripts'/n).is_file() for n in REQUIRED_SCRIPTS),
        'required_tests_found': sum((root/'tests'/n).is_file() for n in REQUIRED_TESTS),
        'required_runs_found': sum((root/'runs'/n).is_dir() for n in RUNS),
        'optional_runs_found': [r for r in OPTIONAL_RUNS if (root/'runs'/r).is_dir()],
        'extra_provenance_files': [relpath(p, root) for p in source_files
                                  if p.name not in REQUIRED_SCRIPTS + REQUIRED_TESTS],
        'possible_workflow11a_producers': attribution_candidates,
        'warnings': ([] if attribution_candidates else [
            'No Workflow 11A producer identified by known output signatures; manually confirm its code path before claiming complete reproducibility.']),
        'missing': missing,
        'symlink_blockers': bad_symlinks,
        'existing_target': (root/'releases'/RELEASE_NAME).exists(),
    }


def run_11h_verify(root: Path) -> dict:
    cmd = [sys.executable, str(root/'scripts'/'run_workflow11h_nucleus_constrained_segmentation.py'), '--verify']
    p = subprocess.run(cmd, cwd=root, text=True, capture_output=True, timeout=240)
    valid_status = bool(re.search(r'"status"\s*:\s*"PASS"', p.stdout))
    return {
        'command': 'python scripts/run_workflow11h_nucleus_constrained_segmentation.py --verify',
        'returncode': p.returncode,
        'stdout': p.stdout[-12000:],
        'stderr': p.stderr[-12000:],
        'status_pass': (p.returncode == 0 and valid_status),
    }


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def copy_checked(src: Path, dst: Path, root: Path):
    if not safe_regular(src, root):
        raise RuntimeError('Refusing missing/symlink/out-of-root source: '+str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    a = sha256(src)
    if sha256(dst) != a:
        raise RuntimeError('Content changed during copying: '+str(src))
    return a


def create_release(root: Path, verify_h: bool):
    pre = preliminary_check(root)
    if pre['missing'] or pre['symlink_blockers']:
        raise RuntimeError('Missing or unsafe required inputs. Run --check: '+str(pre['missing']+pre['symlink_blockers']))
    target = root/'releases'/RELEASE_NAME
    if target.exists() or target.is_symlink():
        raise FileExistsError('Release exists; refusing overwrite: '+str(target))
    validation = run_11h_verify(root) if verify_h else {
        'status_pass': False,
        'skipped': True,
        'reason': '--skip-workflow-verify explicitly requested; release is unverified'}
    if not validation['status_pass'] and verify_h:
        raise RuntimeError('Workflow 11H --verify did not PASS; cannot package verified release. '+str(validation))
    release_parent = target.parent
    release_parent.mkdir(parents=True, exist_ok=True)
    stage = release_parent / ('.'+RELEASE_NAME+'.staging-'+str(os.getpid()))
    if stage.exists():
        raise FileExistsError('Staging directory already exists: '+str(stage))
    stage.mkdir()
    try:
        source_inventory = []
        # Audit exact current run outputs, keeping originals in runs/.
        available = RUNS + [x for x in OPTIONAL_RUNS if (root/'runs'/x).is_dir()]
        for run in available:
            for src in files_inside(root/'runs'/run, root):
                st = src.stat()
                source_inventory.append({
                    'path': relpath(src, root),
                    'size_bytes': st.st_size,
                    'sha256': sha256(src),
                    'role': 'external run output (original retained in runs/)'
                })
        copied = []
        for src in detect_source_files(root):
            dest = stage/'source_snapshot'/src.relative_to(root)
            digest = copy_checked(src, dest, root)
            copied.append({'source': relpath(src, root), 'snapshot': relpath(dest, stage), 'sha256': digest})
        remaining = MAX_TOTAL_COPIES
        run_copies = []
        skipped = []
        # Prioritize corrected current release before legacy exploratory results.
        ordered = (['3d_nucleus_constrained_celllike_segmentation_v0_2',
                    '3d_cadherin_assisted_celllike_segmentation_v0_1'] +
                   [r for r in available if r not in (
                       '3d_nucleus_constrained_celllike_segmentation_v0_2',
                       '3d_cadherin_assisted_celllike_segmentation_v0_1')])
        for run in ordered:
            for src in files_inside(root/'runs'/run, root):
                s = src.stat().st_size
                r = relpath(src, root)
                if src.suffix.lower() not in CURATED_SUFFIXES:
                    skipped.append({'path': r, 'reason': 'not a curated output type'})
                    continue
                if s > MAX_SINGLE_COPY or s > remaining:
                    skipped.append({'path': r, 'reason': 'copy size budget (still in source inventory)'})
                    continue
                dest = stage/'selected_results'/src.relative_to(root)
                copy_checked(src, dest, root)
                run_copies.append({'source': r, 'snapshot': relpath(dest, stage)})
                remaining -= s
        write_json(stage/'SOURCE_OUTPUT_INVENTORY.json', source_inventory)
        write_json(stage/'CURATED_OUTPUT_SELECTION.json', {'copied': run_copies, 'not_copied': skipped,
                   'max_single_copy_bytes': MAX_SINGLE_COPY,
                   'max_total_copies_bytes': MAX_TOTAL_COPIES})
        write_json(stage/'WORKFLOW11H_VERIFICATION.json', validation)
        manifest = {
            'release': RELEASE_NAME,
            'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'status': 'exploratory research prototype, NOT externally validated',
            'workflow11h_verify_pass': bool(validation['status_pass']),
            'source_snapshot': copied,
            'run_directories_inventoried': available,
            'external_output_file_count': len(source_inventory),
            'selected_output_file_count': len(run_copies),
            'original_data_duplicated': False,
            'limitations': [
                'Pseudo-nuclear labels are not validated whole-cell masks.',
                '2D slice-wise candidate segmentation is not 3D cell tracking.',
                'Physical Z spacing not independently established.',
                'No independent manual cell-boundary truth or external specimen validation.',
                'Earlier HER2 improvement pertains to pre-repair v0.1, not the corrected v0.2.',
                'Source snapshots may depend on other files in the main repository; not a standalone install.'
            ]
        }
        write_json(stage/'release_manifest.json', manifest)
        readme = '''# Workflow 11 research snapshot — v0.2 exploratory

**Status:** Candidate serial-section spatial protein attribution / cell-boundary inference research prototype. This is **not** a validated whole-cell segmentation release or publication-ready result.

## Contents
- `source_snapshot/`: versions of Workflow 11 scripts and tests present at release creation, plus selected configs/environment files.
- `selected_results/`: size-capped previews and small tables copied from listed `runs/` folders.
- `SOURCE_OUTPUT_INVENTORY.json`: SHA-256 and relative path for **every** file present in the listed Workflow 11 run directories at packaging time, including large outputs not copied.
- `CURATED_OUTPUT_SELECTION.json`: explicit list of copied and not-copied results.
- `WORKFLOW11H_VERIFICATION.json`: actual result of `python scripts/run_workflow11h_nucleus_constrained_segmentation.py --verify` when packaging.
- `release_manifest.json`: release-level provenance and limitations.
- `SHA256SUMS`: hashes for every archived file (except the checksum list itself).

## Workflow mapping
11A–11B: spatial region signal attribution. 11C: XY shift controls. 11D: boundary robustness. 11E: panCK epithelial-domain feasibility. 11F: Yb174 Cadherin/Ir193/panCK/HER2 visual feasibility. 11G: original 2D Cadherin-assisted vs Voronoi candidate segmentation (retained as failure-analysis baseline). 11H: nucleus-constrained corrected candidate segmentation (current algorithm candidate).

## Core research safeguards
HER2 is **held out of boundary construction**. Yb174 is Cadherin-associated (source panel says E/P Cadherin, antibody clone suggests E-Cadherin); not a complete cell-membrane ground truth. All cell-like labels in Workflow 11G/11H are slice-wise candidate partitions, not confirmed 3D cells. The old +4.25 HER2 interface-core delta and ~90% positive paired results are from **the old, pre-repair Workflow 11G** and must not be attributed to Workflow 11H.

## Data footprint and reproducibility
Primary TIFF/ZIP datasets and large arrays are intentionally **not duplicated**. The manifest uses paths relative to the original project root, so source files remain discoverable. The original `runs/`, `scripts/`, `tests/`, `configs/`, and prior release directories are never changed by this tool. The source snapshot may use modules outside this release; to execute it, retain the original repository and datasets.

## Next scientific validation
Verify pixel-level containment for full pseudo-nuclear seed fragments and equal domain coverage. Annotate a limited number of cell-cell boundaries independently and evaluate against the same baseline, then repeat on additional specimens. Distinguish software integrity (PASS) from biological accuracy (not yet established).

## Verification
From project root:

```bash
python scripts/finalize_workflow11_release.py --verify
python scripts/finalize_workflow11_release.py --verify --verify-sources
```

Second command recomputes hashes of all external run outputs and may take longer.
'''
        (stage/'README.md').write_text(readme, encoding='utf-8')
        checks = []
        for src in files_inside(stage, stage):
            if src.name == 'SHA256SUMS':
                continue
            checks.append(sha256(src)+'  '+relpath(src, stage))
        (stage/'SHA256SUMS').write_text('\n'.join(checks)+'\n',encoding='utf-8')
        # Ensure self-consistency *before* atomic rename.
        audit_release(stage, root, verify_sources=False)
        if target.exists():
            raise FileExistsError('Target was created by another process: '+str(target))
        stage.rename(target)
        return {'status': 'PASS', 'release': str(target), 'snapshot_files': len(copied),
                'run_files_in_inventory':len(source_inventory),'curated_results_copied':len(run_copies),
                'workflow11h_verified': bool(validation['status_pass']),
                'notes':'Project originals unchanged; source TIFFs not duplicated.'}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def audit_release(target: Path, root: Path, verify_sources: bool):
    checksum_file = target/'SHA256SUMS'
    if not checksum_file.is_file():
        raise RuntimeError('SHA256SUMS missing')
    records = 0
    registered = set()
    for line in checksum_file.read_text(encoding='utf-8').splitlines():
        digest, sep, rel = line.partition('  ')
        if sep != '  ' or not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise RuntimeError('Bad checksum record: '+line)
        f = target/rel
        if rel in registered or not safe_regular(f, target) or sha256(f) != digest:
            raise RuntimeError('Archive file missing, modified, unsafe or duplicate: '+rel)
        registered.add(rel)
        records += 1
    expected = {relpath(f, target) for f in files_inside(target, target)} - {'SHA256SUMS'}
    if expected != registered:
        raise RuntimeError('Untracked or missing snapshot files: '+str(expected.symmetric_difference(registered)))
    manifest = json.loads((target/'release_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('release') != RELEASE_NAME:
        raise RuntimeError('Wrong manifest release')
    checked_src = 0
    if verify_sources:
        for x in json.loads((target/'SOURCE_OUTPUT_INVENTORY.json').read_text(encoding='utf-8')):
            src = root/x['path']
            if not safe_regular(src, root) or sha256(src) != x['sha256']:
                raise RuntimeError('Original run source missing or changed: '+x['path'])
            checked_src += 1
    return {'status':'PASS', 'release':str(target), 'archived_files_verified':records,
            'original_run_sources_verified':checked_src,
            'workflow11h_verified_at_packaging':manifest.get('workflow11h_verify_pass',False)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--verify', action='store_true')
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--verify-sources', action='store_true', help='Also check hashes of original run output files (slow)')
    p.add_argument('--skip-workflow-verify', action='store_true', help='EXPERIMENTAL: package even if upstream 11H --verify unavailable; mark release unverified')
    args = p.parse_args()
    root = args.project_root.resolve()
    if not root.is_dir():
        p.error('Project root not found: '+str(root))
    try:
        if args.check:
            info = preliminary_check(root)
            print(json.dumps(info, indent=2, ensure_ascii=False))
            if info['missing'] or info['symlink_blockers']:
                return 2
            return 0
        if args.apply:
            result = create_release(root, verify_h=not args.skip_workflow_verify)
        else:
            result = audit_release(root/'releases'/RELEASE_NAME, root, args.verify_sources)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({'status':'FAIL', 'reason':str(e)}, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
