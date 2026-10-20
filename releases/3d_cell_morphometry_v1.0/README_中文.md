# Workflow 07 — 3D Cell Morphometry v1.0 — Archived release

**Status:** Complete / source and results verified. `snapshot/` is archival; **do not execute the snapshot as a second live project**.

## Canonical project layout

- Production entrypoint: `scripts/run_3d_cell_morphometry.py`
- Implementation: `scripts/cell_morphometry_core.py`
- Tests: `tests/test_cell_morphometry_v1.py`
- Documentation: `docs/CELL_MORPHOMETRY_V1_中文.md`
- Unmodified results: `runs/3d_cell_morphometry_v1/`
- Version record: `releases/3d_cell_morphometry_v1.0/`

## Verified reference statistics

- Source label stack interpreted as Z,Y,X: [16, 1205, 1215]; TIFF axis metadata alone cannot verify registration.
- Author watershed pseudo instances: 12,663
- Foreground voxels: 1,471,040
- Z boundary IDs: 3,447
- 6-connected multiple component IDs: 1
- QC review flagged: 3,448

This is a single specimen with *automated pseudo labels*. No verified voxel physical spacing or biological cell identity, no per-instance protein integration, and no generalization claim.

## Commands from project root

```bash
python tools/finalize_workflow07.py --verify
python -m pytest -q tests/test_cell_morphometry_v1.py
bash releases/3d_cell_morphometry_v1.0/run_new.sh runs/3d_cell_morphometry_v1_repeat
```

New runs must use a **new output path**; do not overwrite v1 results. Historical organizer and duplicate workflow directory can be restored from `_archive/workflow07_obsolete_files/`.
