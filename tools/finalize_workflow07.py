#!/usr/bin/env python3
"""Consolidate Workflow 07 under releases/ and remove obsolete duplicates from the active tree.

Dry-run by default. Releases, source TIFF, runs and canonical Morphometry scripts remain intact.
Apply quarantines only an older duplicate workflow index and the obsolete organizer files,
never the historical v1 segmentation scripts and never any module with detected references.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

REL = Path('releases/3d_cell_morphometry_v1.0')
OLD = Path('workflows/07_3d_cell_morphometry')
SEG_REL = Path('releases/3d_instance_segmentation_v1.0')
RUN = Path('runs/3d_cell_morphometry_v1')
ARCHIVE = Path('_archive/workflow07_obsolete_files')
SOURCES = {
    'scripts/cell_morphometry_core.py': 'Morphometry measurement / QC core',
    'scripts/run_3d_cell_morphometry.py': 'Single supported execution entrypoint',
    'tests/test_cell_morphometry_v1.py': '3D morphometry synthetic tests',
    'docs/CELL_MORPHOMETRY_V1_中文.md': 'Workflow07 methodology and caveats',
}
RESULTS = ('cell_morphometry.csv','cell_shape_features.csv','cell_slice_profiles.csv',
           'cell_morphology_qc.csv','morphometry_summary.json','output_file_sha256.json')
OBSOLETE = (Path('tools/organize_workflow07.py'), Path('tests/test_organize_workflow07.py'),
            Path('docs/WORKFLOW07_ORGANIZATION_中文.md'))


def digest(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024*4),b''):
            h.update(block)
    return h.hexdigest()


def get_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


def check_regular(p):
    if not p.is_file() or p.is_symlink():
        raise RuntimeError(f'Missing or unsafe file: {p}')


def relsafe(s):
    if not isinstance(s,str):
        raise RuntimeError('Manifest path must be a string')
    p = PurePosixPath(s)
    if p.is_absolute() or not p.parts or '..' in p.parts or p.parts[0] in ('.',''):
        raise RuntimeError(f'Unsafe manifest path: {s}')
    return Path(*p.parts)


def check_prior_release(root):
    p = root/SEG_REL/'MANIFEST_SHA256.csv'
    check_regular(p)
    with p.open(newline='',encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'file','sha256'} <= set(rows[0]):
        raise RuntimeError('Invalid prior release SHA manifest')
    for row in rows:
        source = root/SEG_REL/relsafe(row['file'])
        check_regular(source)
        if digest(source) != row['sha256']:
            raise RuntimeError(f'Prior release changed: {source}')
    return len(rows)


def check_inputs(root):
    for rel in SOURCES:
        check_regular(root/rel)
    run = root/RUN
    for name in RESULTS:
        check_regular(run/name)
    summary = get_json(run/'morphometry_summary.json')
    if summary.get('version') != '1.0.0':
        raise RuntimeError('Expected Morphometry v1.0.0 result summary')
    if digest(root/'scripts/cell_morphometry_core.py') != summary['core_code_sha256']:
        raise RuntimeError('Core source no longer matches the v1.0 run')
    orig = Path(summary['source_mask_path'])
    portable = root/'data/inspect/LVILymph'/orig.name
    if not orig.is_file():
        orig = portable
    check_regular(orig)
    if digest(orig) != summary['source_mask_sha256']:
        raise RuntimeError('Input TIFF has changed')
    output_hash = get_json(run/'output_file_sha256.json')
    if not isinstance(output_hash,dict) or not output_hash:
        raise RuntimeError('Missing output checksums')
    for name in RESULTS[:-1]:
        if name not in output_hash:
            raise RuntimeError(f'Output not covered by prior manifest: {name}')
    for rel, sha in output_hash.items():
        p = run/relsafe(rel)
        check_regular(p)
        if digest(p) != sha:
            raise RuntimeError(f'Morphometry result modified: {rel}')
    return summary, len(output_hash)


def check_old(root):
    old = root/OLD
    if not old.exists():
        return False
    if not old.is_dir() or old.is_symlink():
        raise RuntimeError('Old workflow path is not a regular directory')
    lock_path = old/'workflow07_lock.json'
    check_regular(lock_path)
    old_lock = get_json(lock_path)
    for rel, sha in old_lock['canonical_sha256'].items():
        relsafe(rel)
        if digest(root/rel) != sha:
            raise RuntimeError(f'Old workflow canonical source changed: {rel}')
    for rel, sha in old_lock['snapshot_sha256'].items():
        relsafe(rel)
        if digest(old/'snapshot'/rel) != sha:
            raise RuntimeError(f'Old workflow snapshot changed: {rel}')
    for rel, sha in old_lock['generated_sha256'].items():
        relsafe(rel)
        if digest(old/rel) != sha:
            raise RuntimeError(f'Old workflow documentation changed: {rel}')
    return True


def readme(summary):
    return f'''# Workflow 07 — 3D Cell Morphometry v1.0 — Archived release

**Status:** Complete / source and results verified. `snapshot/` is archival; **do not execute the snapshot as a second live project**.

## Canonical project layout

- Production entrypoint: `scripts/run_3d_cell_morphometry.py`
- Implementation: `scripts/cell_morphometry_core.py`
- Tests: `tests/test_cell_morphometry_v1.py`
- Documentation: `docs/CELL_MORPHOMETRY_V1_中文.md`
- Unmodified results: `runs/3d_cell_morphometry_v1/`
- Version record: `releases/3d_cell_morphometry_v1.0/`

## Verified reference statistics

- Source label stack interpreted as Z,Y,X: {summary['shape_assumed_zyx']}; TIFF axis metadata alone cannot verify registration.
- Author watershed pseudo instances: {summary['n_instance_ids']:,}
- Foreground voxels: {summary['foreground_voxels']:,}
- Z boundary IDs: {summary['n_touching_z_boundary']:,}
- 6-connected multiple component IDs: {summary['n_multicomponent_6']}
- QC review flagged: {summary['n_qc_review_flagged']:,}

This is a single specimen with *automated pseudo labels*. No verified voxel physical spacing or biological cell identity, no per-instance protein integration, and no generalization claim.

## Commands from project root

```bash
python tools/finalize_workflow07.py --verify
python -m pytest -q tests/test_cell_morphometry_v1.py
bash releases/3d_cell_morphometry_v1.0/run_new.sh runs/3d_cell_morphometry_v1_repeat
```

New runs must use a **new output path**; do not overwrite v1 results. Historical organizer and duplicate workflow directory can be restored from `_archive/workflow07_obsolete_files/`.
'''


def build_release(root, summary, target_rel=REL):
    target = root/target_rel
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.wf07_release_tmp_',dir=target.parent))
    try:
        hashes = {}
        for rel in SOURCES:
            src = root/rel
            hashes[rel] = digest(src)
            dest = tmp/'snapshot'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest)
        (tmp/'README_中文.md').write_text(readme(summary),encoding='utf-8')
        with (tmp/'script_workflow_index.csv').open('w',encoding='utf-8',newline='') as f:
            w = csv.writer(f)
            w.writerow(('canonical_relative_path','purpose','sha256'))
            for rel,desc in SOURCES.items():
                w.writerow((rel,desc,hashes[rel]))
        # Avoid second 12k-row dataset inside the release: only summary and canonical checksums.
        audit = {'provenance':summary.get('provenance'),
                 'shape_assumed_zyx':summary['shape_assumed_zyx'],
                 'n_instance_ids':summary['n_instance_ids'],
                 'foreground_voxels':summary['foreground_voxels'],
                 'n_multiplane_ids':summary['n_multiplane_ids'],
                 'n_z_gapped_ids':summary['n_z_gapped_ids'],
                 'n_multicomponent_6':summary['n_multicomponent_6'],
                 'n_multicomponent_26':summary['n_multicomponent_26'],
                 'n_touching_z_boundary':summary['n_touching_z_boundary'],
                 'n_qc_review_flagged':summary['n_qc_review_flagged'],
                 'source_mask_sha256':summary['source_mask_sha256'],
                 'limitations':summary.get('interpretation_limits',[])}
        (tmp/'result_audit_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
        run_script = '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash releases/3d_cell_morphometry_v1.0/run_new.sh runs/NEW_OUTPUT" >&2
  exit 2
fi
cd "$ROOT"
if [[ -e "$1" ]]; then
  echo "Refusing to overwrite existing directory: $1" >&2
  exit 2
fi
python scripts/run_3d_cell_morphometry.py --data-dir data/inspect/LVILymph --out "$1" --assume-zyx
'''
        verify_script = '''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python tools/finalize_workflow07.py --verify
python -m pytest -q tests/test_cell_morphometry_v1.py
'''
        (tmp/'run_new.sh').write_text(run_script,encoding='utf-8')
        (tmp/'verify.sh').write_text(verify_script,encoding='utf-8')
        for name in ('run_new.sh','verify.sh'):
            (tmp/name).chmod(0o755)
        generated = ['README_中文.md','script_workflow_index.csv','result_audit_summary.json','run_new.sh','verify.sh']
        lock = {'version':'1.0.0','type':'workflow07-frozen-index','canonical_sha256':hashes,
                'snapshot_sha256':{k:digest(tmp/'snapshot'/k) for k in hashes},
                'source_mask_sha256':summary['source_mask_sha256'],
                'result_manifest_sha256':digest(root/RUN/'output_file_sha256.json'),
                'result_summary_sha256':digest(root/RUN/'morphometry_summary.json'),
                'generated_sha256':{name:digest(tmp/name) for name in generated}}
        (tmp/'workflow07_lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2),encoding='utf-8')
        if target.exists():
            raise RuntimeError('Release path already exists; refusing overwrite')
        tmp.rename(target)
        return target
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def verify_release(root):
    target = root/REL
    if not target.is_dir() or target.is_symlink():
        raise RuntimeError('Workflow07 release directory not found/unsafe')
    lock = get_json(target/'workflow07_lock.json')
    if lock.get('type') == 'workflow07-frozen-index':
        for rel, expected in lock['canonical_sha256'].items():
            relsafe(rel)
            check_regular(root/rel)
            if digest(root/rel) != expected:
                raise RuntimeError(f'Canonical code changed: {rel}')
        for rel, expected in lock['snapshot_sha256'].items():
            relsafe(rel)
            check_regular(target/'snapshot'/rel)
            if digest(target/'snapshot'/rel) != expected:
                raise RuntimeError(f'Snapshot modified: {rel}')
        for rel, expected in lock['generated_sha256'].items():
            relsafe(rel)
            check_regular(target/rel)
            if digest(target/rel) != expected:
                raise RuntimeError(f'Release metadata modified: {rel}')
        summary = get_json(root/RUN/'morphometry_summary.json')
        if digest(root/RUN/'morphometry_summary.json') != lock['result_summary_sha256']:
            raise RuntimeError('Morphometry summary changed')
        if digest(root/RUN/'output_file_sha256.json') != lock['result_manifest_sha256']:
            raise RuntimeError('Morphometry output manifest changed')
        if summary['source_mask_sha256'] != lock['source_mask_sha256']:
            raise RuntimeError('Source mask hash differs from release')
    else:
        # Support copies created manually from the earlier organizer.
        if lock.get('version') != '1.0.0':
            raise RuntimeError('Unrecognized Workflow07 lock')
        for label, base in [('canonical_sha256',root),('snapshot_sha256',target/'snapshot'),('generated_sha256',target)]:
            for rel, expected in lock[label].items():
                relsafe(rel)
                check_regular(base/rel)
                if digest(base/rel) != expected:
                    raise RuntimeError(f'Copied old release hash mismatch: {rel}')
        if digest(root/RUN/'output_file_sha256.json') != lock['original_output_manifest_sha256']:
            raise RuntimeError('Original output changed')
        if digest(root/RUN/'morphometry_summary.json') != lock['original_summary_sha256']:
            raise RuntimeError('Original morphometry summary changed')
        raise RuntimeError('Old-style copied Workflow07 release: safe copy detected but uses stale workflows/ paths. Refusing cleanup; obtain migrated release first.')
    check_inputs(root)
    print(f'PASS: Workflow07 release verified: {target}')


def validate_oldstyle_release_copy(root):
    """Verify a manually copied old organizer index before replacing it with a path-correct release."""
    base = root/REL
    if not base.is_dir() or base.is_symlink():
        raise RuntimeError('Legacy release path is not a regular directory')
    lock = get_json(base/'workflow07_lock.json')
    if lock.get('version') != '1.0.0' or 'canonical_sha256' not in lock:
        raise RuntimeError('Unknown existing release; refusing any migration')
    for label, folder in (('canonical_sha256',root),('snapshot_sha256',base/'snapshot'),('generated_sha256',base)):
        for rel, expected in lock[label].items():
            relsafe(rel)
            check_regular(folder/rel)
            if digest(folder/rel) != expected:
                raise RuntimeError(f'Original copied release modified: {rel}')
    if digest(root/RUN/'output_file_sha256.json') != lock['original_output_manifest_sha256']:
        raise RuntimeError('Copied release input output manifest differs')
    if digest(root/RUN/'morphometry_summary.json') != lock['original_summary_sha256']:
        raise RuntimeError('Copied release input summary differs')
    return True


def migrate_oldstyle_release(root, summary):
    """Atomically replace old-style copied index; retain untouched original in a recoverable archive."""
    validate_oldstyle_release_copy(root)
    batch = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
    staging_rel = REL.parent/('.workflow07_upgrade_'+uuid.uuid4().hex[:8])
    staging = build_release(root,summary,staging_rel)
    legacy = root/REL
    dest = root/ARCHIVE/batch/'files'/REL
    dest.parent.mkdir(parents=True,exist_ok=True)
    moved = False
    created = False
    try:
        legacy.rename(dest)
        moved = True
        staging.rename(legacy)
        created = True
        verify_release(root)
        (root/ARCHIVE/batch/'quarantine_manifest.json').write_text(json.dumps({
            'version':1,'entries':[{'path':REL.as_posix(),'directory':True}],
            'note':'BACKUP of original old-style copied release. New corrected release occupies its original path; restore requires first moving the new release away.'
        }, indent=2),encoding='utf8')
    except Exception:
        if created and legacy.exists():
            shutil.rmtree(legacy)
        if moved and dest.exists():
            dest.rename(legacy)
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(f'UPGRADED existing copied release; old version safely backed up: {dest}')
    return batch


def nonactive_references(root, candidate:Path, excluded:set[Path]):
    # Conservative literal-name scan: active Python/scripts/config files, exclude archived/read-only snapshots.
    tokens = (candidate.name, candidate.stem)
    matches = []
    for base in (root/'scripts',root/'tests',root/'tools',root/'docs'):
        if not base.is_dir():
            continue
        for p in base.rglob('*'):
            if not p.is_file() or p.is_symlink() or p.suffix not in ('.py','.sh','.toml','.yaml','.yml'):
                continue
            relative = p.relative_to(root)
            if relative in excluded or relative in {Path('tools/finalize_workflow07.py'), Path('tests/test_finalize_workflow07.py')}:
                continue
            try:
                content = p.read_text(encoding='utf-8')
                if any(token in content for token in tokens):
                    matches.append(relative.as_posix())
            except UnicodeDecodeError:
                return ['UNREADABLE active file: '+str(relative)]
    return matches


def cleanup_candidates(root):
    exclusions = {p for p in OBSOLETE if (root/p).is_file()}
    found = []
    if (root/OLD).is_dir():
        check_old(root)
        found.append(OLD)
    for p in OBSOLETE:
        obj = root/p
        if obj.exists():
            if not obj.is_file() or obj.is_symlink():
                raise RuntimeError(f'Unsafe candidate path: {p}')
            matches = nonactive_references(root,p,exclusions)
            if matches:
                print(f'KEEP: {p}: referenced by {", ".join(matches[:5])}')
                continue
            found.append(p)
    return found


def quarantine(root, candidates):
    if not candidates:
        return None
    label = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
    dest = root/ARCHIVE/label
    entries = []
    try:
        (dest/'files').mkdir(parents=True,exist_ok=False)
        for rel in candidates:
            src = root/rel
            dst = dest/'files'/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists() or src.is_symlink():
                raise RuntimeError('Unsafe quarantine target')
            shutil.move(str(src),str(dst))
            entries.append({'path':rel.as_posix(), 'directory':dst.is_dir()})
        (dest/'quarantine_manifest.json').write_text(json.dumps({'version':1,'entries':entries},indent=2),encoding='utf-8')
        # Eliminate the redundant, now empty workflows/ folder, but preserve unrelated workflows.
        w = root/'workflows'
        if w.is_dir() and not any(w.iterdir()):
            w.rmdir()
        return label
    except Exception:
        for entry in reversed(entries):
            old = dest/'files'/entry['path']
            new = root/entry['path']
            new.parent.mkdir(parents=True,exist_ok=True)
            if old.exists() and not new.exists():
                shutil.move(str(old),str(new))
        raise


def restore(root, label):
    if not re.fullmatch(r'\d{8}T\d{6}Z_[0-9a-f]{6}',label):
        raise RuntimeError('Invalid batch id')
    base = root/ARCHIVE/label
    path = base/'quarantine_manifest.json'
    check_regular(path)
    entries = get_json(path)['entries']
    for entry in entries:
        rel = relsafe(entry['path'])
        source = base/'files'/rel
        dest = root/rel
        if not source.exists() or dest.exists():
            raise RuntimeError(f'Restore conflict or missing source: {rel}')
    for entry in entries:
        rel = relsafe(entry['path'])
        source = base/'files'/rel
        dest = root/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(source),str(dest))
    print(f'RESTORED: {len(entries)} entries from {label}')


def purge(root, label, confirm):
    if confirm != 'DELETE_QUARANTINED_WORKFLOW07':
        raise RuntimeError('Purge requires --confirm DELETE_QUARANTINED_WORKFLOW07')
    if not re.fullmatch(r'\d{8}T\d{6}Z_[0-9a-f]{6}',label):
        raise RuntimeError('Invalid batch id')
    base = root/ARCHIVE/label
    check_regular(base/'quarantine_manifest.json')
    verify_release(root)
    shutil.rmtree(base)
    print(f'PERMANENTLY REMOVED: {label}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--apply',action='store_true',help='Create release if absent; quarantine obsolete duplicate index/files')
    grp.add_argument('--verify',action='store_true',help='Verify release and original input/output checksums')
    grp.add_argument('--restore',metavar='BATCH_ID',help='Recover quarantined files')
    grp.add_argument('--purge',metavar='BATCH_ID',help='Permanently delete quarantine batch')
    ap.add_argument('--confirm',default='')
    ap.add_argument('--no-tests',action='store_true',help='Skip pytest subprocess when applying (NOT recommended)')
    a=ap.parse_args(argv)
    root=a.project.resolve()
    try:
        if a.restore:
            restore(root,a.restore)
            return 0
        if a.purge:
            purge(root,a.purge,a.confirm)
            return 0
        if a.verify:
            check_prior_release(root)
            verify_release(root)
            return 0
        count = check_prior_release(root)
        state,n_results = check_inputs(root)
        old = check_old(root)
        release = root/REL
        print(f'PASS: segmentation release {count} hashes; Morphometry output {n_results} checksums')
        if not release.exists():
            print(f'NEW RELEASE: {REL}')
        else:
            # Existing manually copied legacy release: need explicit migration handling.
            lock = get_json(release/'workflow07_lock.json')
            if lock.get('type') != 'workflow07-frozen-index':
                validate_oldstyle_release_copy(root)
                print(f'OLD-STYLE release copy detected in {REL}: valid, but stale workflows/ paths')
                if not a.apply:
                    print('DRY-RUN: --apply will back up the original release, then create a corrected release at the same path')
                    return 0
            else:
                verify_release(root)
        candidates = cleanup_candidates(root)
        print(f'OLD workflow index present: {old}; cleanup candidates: {len(candidates)}')
        for path in candidates:
            print('  QUARANTINE',path)
        if not a.apply:
            print('DRY-RUN. Apply with --apply. Raw data, checkpoints, runs and all releases are protected.')
            return 0
        if not a.no_tests:
            subprocess.run([sys.executable,'-m','pytest','-q','tests/'],cwd=root,check=True)
        made = False
        if not release.exists():
            build_release(root,state)
            made=True
        elif get_json(release/'workflow07_lock.json').get('type') != 'workflow07-frozen-index':
            migrate_oldstyle_release(root,state)
            made=True
        verify_release(root)
        label = quarantine(root,candidates)
        try:
            if not a.no_tests:
                subprocess.run([sys.executable,'-m','pytest','-q','tests/'],cwd=root,check=True)
            verify_release(root)
        except Exception:
            if label:
                restore(root,label)
            raise
        if label:
            print(f'QUARANTINED {len(candidates)} obsolete entries: {root/ARCHIVE/label}')
            print(f'UNDO: python tools/finalize_workflow07.py --restore {label}')
        else:
            print('No obsolete organizer files remaining')
        print(f'DONE: {REL}; Release created: {made}; canonical scripts unchanged')
        return 0
    except (RuntimeError,FileNotFoundError,OSError,KeyError,subprocess.CalledProcessError) as e:
        print('ABORT:',e,file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
