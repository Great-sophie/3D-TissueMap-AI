#!/usr/bin/env python3
"""Fail-closed export audit; allows ONLY three known aggregate result PNGs."""
from pathlib import Path
import re
import sys
import struct

BAD_SUFFIXES = {'.tif','.tiff','.zip','.7z','.h5','.hdf5','.pt','.pth','.ckpt','.npy','.npz','.parquet','.nii','.pem','.key','.p12'}
BAD_DIRS = {'data','runs','_archive','checkpoints','raw','models','outputs','node_modules','.venv'}
MAX_BYTES = 1024*1024
VIEWER_LIMITS = {
    'docs/viewers/roi_voxel_viewer.html': 20*1024*1024,
    'docs/viewers/centroid_marker_viewer.html': 20*1024*1024,
}

PERMITTED_PNG = {
    'assets/results/z_sampling.png',
    'assets/results/transfer_auroc.png',
    'assets/results/primary_ablation.png',
}
SECRETS = [
    ('credential',re.compile(r'\b(?:ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{30,}\b|github_pat_[A-Za-z0-9_]{35,}|\bsk-[A-Za-z0-9_-]{24,}\b')),
    ('private_key',re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
    ('credential_assignment',re.compile(r'''(?im)^\s*(?:api[_-]?key|client[_-]?secret|access[_-]?token|password|passwd)\s*[:=]\s*['\"][^'\"]{10,}['\"]''')),
]
USER_PATH = re.compile(r'/home/[^/\s\'\"]+|/media/[^/\s\'\"]+|[A-Za-z]:\\\\Users\\\\[^\\\\\s\'\"]+', re.I)

def clean_png(path):
    """Check signature, IHDR/dimensions, safe size. Does not assert image contents are privacy-safe."""
    if path.stat().st_size > 512*1024:
        return False
    with path.open('rb') as f:
        head = f.read(24)
    if len(head)<24 or head[:8] != b'\x89PNG\r\n\x1a\n' or head[12:16] != b'IHDR':
        return False
    width, height = struct.unpack('>II',head[16:24])
    return 100<=width<=6000 and 100<=height<=6000

def main():
    root=Path.cwd()
    violations=[];n=0
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '.git' in path.parts or '__pycache__' in path.parts:
            continue
        rel=path.relative_to(root)
        name=rel.as_posix()
        n+=1
        if path.is_symlink() or any(s in BAD_DIRS for s in rel.parts[:-1]) or path.suffix.lower() in BAD_SUFFIXES or path.stat().st_size>VIEWER_LIMITS.get(name,MAX_BYTES) or path.name.startswith('.env'):
            violations.append((name,'restricted file type, symlink, directory, or >1 MB'))
            continue
        if path.suffix.lower()=='.png':
            if name not in PERMITTED_PNG or not clean_png(path):
                violations.append((name,'PNG outside aggregate figure whitelist / invalid'))
            continue
        try:
            content=path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            violations.append((name,'non-UTF8/binary file'))
            continue
        for kind, pattern in SECRETS:
            if pattern.search(content):violations.append((name,kind))
        if name not in {'docs/REPRODUCIBILITY.md','docs/EXPORT_REVIEW.md','prepush_audit.py'} and USER_PATH.search(content):
            violations.append((name,'local absolute path'))
    print(f'Audited {n} GitHub export files')
    if violations:
        print('BLOCKED: fix the following files before pushing:')
        for name,reason in violations[:80]: print(f'  {name}: {reason}')
        return 2
    print('PASS: All three permitted PNG figures and code/docs passed file and credential checks.')
    print('Manual content and data-rights review is still required.')
    return 0

if __name__=='__main__':sys.exit(main())
