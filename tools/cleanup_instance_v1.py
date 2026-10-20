#!/usr/bin/env python3
"""Conservative cleanup after immutable 3D TissueMap instance-segmentation v1.0 freeze.

Dry-run by default. No original data/models/runs/release contents removed.
--apply quarantines safe caches/backups; with --include-legacy-scripts also archives
unreferenced, hash-identical diagnostic entrypoints. Restore before any purge.
"""
from __future__ import annotations
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import re
import uuid

RELEASE_NAME = '3d_instance_segmentation_v1.0'
ARCHIVE_BASE = Path('_archive') / 'instance_segmentation_v1_cleanup'
CACHE_DIRS = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.ipynb_checkpoints'}
SOURCE_DIRS = ('scripts', 'tests', 'tools')
LEGACY_PATTERNS = (
    re.compile(r'^audit_.*\.py$'),
    re.compile(r'^.*(?:_pilot|_diagnostic|_diagnostics|_fixed_patch_check)\.py$'),
)
CORE_KEEP = {
    'train_segmentation.py', 'train_voxel_affinity.py',
    'instance_watershed_baseline.py', 'train_region_merge_classifier.py',
    'build_region_interface_dataset.py', 'evaluate_learned_region_graph.py',
    'evaluate_z_continuity_rescue.py', 'evaluate_dynamic_region_graph.py',
    'evaluate_affinity_region_graph.py', 'train_center_boundary.py',
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def repo_rel(project: Path, p: Path) -> str:
    return p.relative_to(project).as_posix()


def verify_freeze(project: Path) -> tuple[Path, dict[str, str]]:
    release = project / 'releases' / RELEASE_NAME
    manifest = release / 'MANIFEST_SHA256.csv'
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeError('Verified v1.0 release manifest missing; stop cleanup.')
    copies = {}
    count = 0
    with manifest.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            rel = Path(row['file'])
            if rel.is_absolute() or '..' in rel.parts:
                raise RuntimeError('Unsafe release manifest path.')
            path = release / rel
            if not path.is_file() or path.is_symlink() or sha256(path) != row['sha256']:
                raise RuntimeError(f'Release integrity mismatch: {rel}')
            copies[rel.as_posix()] = row['sha256']
            count += 1
    if count == 0:
        raise RuntimeError('Empty release manifest.')
    return release, copies


def dependencies(project: Path) -> dict[str, set[str]]:
    """Conservatively identify .py filename/module references in active source files.

    Also treat literal filename occurrences in source strings as references.
    Does not prove dynamic imports are absent: that is why quarantine is reversible.
    """
    all_py = []
    for dirname in SOURCE_DIRS:
        d = project / dirname
        if d.is_dir():
            all_py.extend(p for p in d.rglob('*.py') if p.is_file() and not p.is_symlink())
    all_py.extend(p for p in project.glob('*.py') if p.is_file() and not p.is_symlink())
    scripts_dir = project / 'scripts'
    target = {p.stem: p.name for p in scripts_dir.glob('*.py') if p.is_file() and not p.is_symlink()}
    referenced = {n: set() for n in target.values()}
    for p in all_py:
        try:
            code = p.read_text(encoding='utf-8')
            tree = ast.parse(code)
        except (UnicodeDecodeError, SyntaxError, OSError):
            # Unknown parse state: preserve everything rather than take a risk.
            for v in referenced.values():
                v.add(f'UNPARSEABLE:{repo_rel(project,p)}')
            continue
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(x.name.split('.')[0] for x in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split('.')[0])
        for module, filename in target.items():
            if p.parent == scripts_dir and p.name == filename:
                continue
            if module in names or filename in code or f'scripts/{filename}' in code:
                referenced[filename].add(repo_rel(project, p))
    # Direct executable references in non-Python scripts/config, not within frozen releases.
    for dirname in SOURCE_DIRS:
        folder = project / dirname
        if not folder.is_dir():
            continue
        for p in folder.rglob('*'):
            if not p.is_file() or p.is_symlink() or p.suffix not in {'.sh', '.toml', '.yaml', '.yml', '.json'}:
                continue
            if any(part in CACHE_DIRS for part in p.parts):
                continue
            try:
                body = p.read_text(encoding='utf-8')
            except (UnicodeDecodeError, OSError):
                continue
            for filename in target.values():
                if filename in body:
                    referenced[filename].add(repo_rel(project, p))
    return referenced


def scan(project: Path, copies: dict[str, str], include_legacy_scripts: bool):
    items = []
    excluded = []
    seen = set()
    # Scope deliberately excludes runs/, data/, releases/ and .git.
    for dirname in SOURCE_DIRS:
        folder = project / dirname
        if not folder.is_dir() or folder.is_symlink():
            continue
        for p in folder.rglob('*'):
            if any(part in CACHE_DIRS for part in p.relative_to(folder).parts):
                continue
            if p.is_file() and not p.is_symlink() and (p.suffix == '.pyc' or p.name.endswith(('~', '.bak', '.orig'))):
                rel = repo_rel(project, p)
                if rel not in seen:
                    items.append((rel, 'cache_or_backup'))
                    seen.add(rel)
        for p in [folder, *folder.rglob('*')]:
            if p.is_dir() and p.name in CACHE_DIRS and not p.is_symlink():
                rel = repo_rel(project, p)
                if not any(rel.startswith(old + '/') for old in seen):
                    items.append((rel, 'cache_dir'))
                    seen.add(rel)
    for p in project.iterdir():
        if p.is_dir() and p.name in CACHE_DIRS and not p.is_symlink():
            items.append((repo_rel(project, p), 'cache_dir'))
    deps = dependencies(project)
    folder = project / 'scripts'
    if folder.is_dir() and include_legacy_scripts:
        for p in sorted(folder.glob('*.py')):
            if p.is_symlink() or not p.is_file():
                continue
            rel = repo_rel(project, p)
            if p.name in CORE_KEEP:
                excluded.append((rel, 'core entrypoint'))
            elif not any(r.match(p.name) for r in LEGACY_PATTERNS):
                excluded.append((rel, 'not in conservative diagnostic naming class'))
            elif deps.get(p.name):
                excluded.append((rel, 'referenced by: ' + ','.join(sorted(deps[p.name])[:4])))
            elif f'snapshot/{rel}' not in copies:
                excluded.append((rel, 'not included in frozen snapshot'))
            elif sha256(p) != copies[f'snapshot/{rel}']:
                excluded.append((rel, 'changed since freeze, preserve current source'))
            else:
                items.append((rel, 'unreferenced_frozen_diagnostic'))
    return sorted(set(items)), excluded


def apply_quarantine(project: Path, items):
    if not items:
        print('No cleanup candidates. Nothing changed.')
        return
    label = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
    base = project / ARCHIVE_BASE / label
    moved = []
    try:
        base.mkdir(parents=True)
        for rel, reason in items:
            src = project / rel
            if not src.exists() or src.is_symlink():
                print(f'SKIP unavailable / symlink: {rel}')
                continue
            dst = base / 'files' / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise RuntimeError(f'Already exists: {dst}')
            # Safety: all moves remain within project, no symlink paths permitted.
            if any(part.is_symlink() for part in (src, *src.parents) if part != project.parent):
                raise RuntimeError(f'Symlink component: {src}')
            shutil.move(str(src), str(dst))
            moved.append({'relative_path': rel, 'reason': reason})
        (base / 'cleanup_manifest.json').write_text(json.dumps({
            'release': RELEASE_NAME,
            'timestamp_utc': label,
            'items': moved,
            'note': 'RESTORE with --restore <timestamp>; do not use as a deployable project'
        }, indent=2, ensure_ascii=False), encoding='utf-8')
    except Exception:
        for item in reversed(moved):
            source = base / 'files' / item['relative_path']
            target = project / item['relative_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.exists() and not target.exists():
                shutil.move(str(source), str(target))
        raise
    print(f'QUARANTINED: {len(moved)} entries into {base}')
    print(f'Restore: python tools/cleanup_instance_v1.py --restore {label}')


def restore(project: Path, release: Path, label: str):
    if not re.fullmatch(r'\d{8}T\d{6}Z_[0-9a-f]{6}', label):
        raise RuntimeError('Invalid cleanup batch ID.')
    base = project / ARCHIVE_BASE / label
    manifest = base / 'cleanup_manifest.json'
    if not manifest.is_file():
        raise RuntimeError('No quarantine manifest.')
    entries = json.loads(manifest.read_text(encoding='utf-8'))['items']
    collisions = [x['relative_path'] for x in entries if (project / x['relative_path']).exists()]
    if collisions:
        raise RuntimeError(f'Restore aborted: live-path conflicts: {collisions[:10]}')
    for x in entries:
        source = base / 'files' / x['relative_path']
        dest = project / x['relative_path']
        if not source.exists():
            raise RuntimeError(f'Archive file missing: {source}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
    print(f'RESTORED: {len(entries)} paths. Archive report kept at {base}')


def purge(project: Path, label: str, phrase: str):
    if phrase != 'DELETE_ARCHIVED_CLEANUP':
        raise RuntimeError('Purge requires --confirm DELETE_ARCHIVED_CLEANUP')
    if not re.fullmatch(r'\d{8}T\d{6}Z_[0-9a-f]{6}', label):
        raise RuntimeError('Invalid cleanup batch ID.')
    base = project / ARCHIVE_BASE / label
    if not (base / 'cleanup_manifest.json').is_file() or base.is_symlink():
        raise RuntimeError('No eligible cleanup batch.')
    shutil.rmtree(base)
    print(f'PERMANENTLY DELETED archived cleanup batch: {label}')


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--project', type=Path, default=Path.cwd())
    a.add_argument('--apply', action='store_true')
    a.add_argument('--include-legacy-scripts', action='store_true', help='Also quarantine unreferenced, frozen diagnostic scripts')
    a.add_argument('--restore', metavar='BATCH_ID')
    a.add_argument('--purge', metavar='BATCH_ID')
    a.add_argument('--confirm', default='')
    args = a.parse_args(argv)
    project = args.project.expanduser().resolve()
    if not (project / 'scripts').is_dir():
        raise SystemExit('Run from project root containing scripts/.')
    try:
        release, copies = verify_freeze(project)
        if args.restore:
            restore(project, release, args.restore)
            return
        if args.purge:
            purge(project, args.purge, args.confirm)
            return
        items, excluded = scan(project, copies, args.include_legacy_scripts)
        print(f'PASS: Frozen release SHA-256 verified ({len(copies)} manifest entries).')
        print(f'Candidates: {len(items)}; script candidate mode: {args.include_legacy_scripts}')
        for path, kind in items:
            print(f'  {kind:33} {path}')
        if args.include_legacy_scripts:
            print(f'Protected or excluded scripts: {len(excluded)}')
            for path, reason in excluded:
                print(f'  KEEP {path}: {reason}')
        if args.apply:
            apply_quarantine(project, items)
        else:
            print('DRY RUN ONLY. To quarantine, repeat with --apply. Nothing changed.')
    except (RuntimeError, OSError) as exc:
        raise SystemExit(f'ABORT: {exc}')


if __name__ == '__main__':
    main()
