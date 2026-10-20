# Workflow 11 research snapshot — v0.2 exploratory

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
