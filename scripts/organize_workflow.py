#!/usr/bin/env python3
"""Safely migrate old time-based names into a workflow-based 3D-TissueMap AI project.

Run from the existing project root AFTER extracting the workflow update:
    python scripts/organize_workflow.py --apply

No raw data are deleted or overwritten. Older scripts and historical results
are preserved under archive/legacy and runs/legacy respectively. By default,
prints a dry-run plan; --apply performs the moves.
"""
from pathlib import Path
import argparse

ROOT=Path(__file__).resolve().parents[1]
LEGACY=(
 ('scripts/day2_real_training.py','scripts/train_segmentation.py'),
 ('scripts/lvilymph_registration_qc.py','scripts/registration_qc.py'),
 ('scripts/run_three_day_demo.sh','scripts/run_synthetic_demo.sh'),
 ('tests/test_day2_real.py','tests/test_segmentation_training.py'),
 ('tests/test_day1.py','tests/test_volume_qc.py'),
 ('src/tissuemap/day1.py','src/tissuemap/volume_qc.py'),
 ('qc_real3d.py','scripts/inspect_real_imc.py'),
 ('DAY1_三维病理实战.md','docs/01_volume_and_axes_qc.md'),
 ('DAY2_真实数据训练_中文.md','docs/03_segmentation_baselines.md'),
 ('REAL_DATA_QC_中文.md','docs/02_registration_and_labels_qc.md'),
)
OLD_RUNS=(
 'day1_demo','day1_real','lvilymph_day2',
 'lvilymph_real_qc','lvilymph_registration_qc',
)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='move old paths after checking new functional replacements')
    args=parser.parse_args()
    if not (ROOT/'pyproject.toml').exists() or not (ROOT/'src/tissuemap').exists():
        raise SystemExit('Must run from inside the existing project installation')
    actions=[]
    for src, successor in LEGACY:
        old=ROOT/src
        if not old.exists():continue
        if not (ROOT/successor).exists():
            print('SKIP (replacement missing):', src, '->',successor)
            continue
        dest=ROOT/'archive/legacy'/src
        if dest.exists():
            print('SKIP (archive path exists):', dest.relative_to(ROOT))
            continue
        actions.append((old,dest))
    for name in OLD_RUNS:
        old=ROOT/'runs'/name
        if old.exists():
            dest=ROOT/'runs/legacy'/name
            if not dest.exists():actions.append((old,dest))
    for src,dest in actions:
        print(('MOVE' if args.apply else 'WOULD MOVE'), src.relative_to(ROOT),'->',dest.relative_to(ROOT))
        if args.apply:
            dest.parent.mkdir(parents=True,exist_ok=True)
            src.rename(dest)
    if args.apply:
        for folder in ['docs','runs/data_audit','runs/registration_qc','runs/segmentation_baselines','runs/spatial_morphology','runs/molecular_integration','runs/evaluation']:
            (ROOT/folder).mkdir(parents=True,exist_ok=True)
        print('Workflow migration complete. Raw data untouched; old outputs preserved in runs/legacy.')
    else:
        print('Dry run only; rerun with --apply to carry out the listed moves.')

if __name__=='__main__':main()
