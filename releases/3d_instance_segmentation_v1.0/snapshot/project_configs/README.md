# 3D-TissueMap AI

**A reproducible prototype for 3D tissue image segmentation, physically grounded cell morphology, and exploratory spatial proteomics.**

**Status:** functioning codebase; synthetic integration test completed. **No real 3D-IMC patient data have yet been ingested or validated.** This is a research/portfolio prototype, not a clinical device or a novel published segmentation method.

## Motivation

Research question: *What additional measurements become possible when multiplexed tissue imaging is analysed as a 3D volume instead of independent 2D sections?*

Relevance: biomedical computer vision (2D/3D U-Net), 3D tissue morphology (physical voxel spacing), marker-level multimodal fusion (single-cell protein intensities), robust assessment (sample/patient splits, Dice/IoU/HD95), and research software engineering (CLI, CI, tests).

## Start in 5 minutes (uses synthetic data)

Linux or macOS, Python 3.10+, PyTorch 2.1+, CPU or CUDA:

```bash
cd 3D-TissueMap-AI
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
OMP_NUM_THREADS=2 python -m tissuemap.cli demo --output runs/demo --epochs 10
```

Example files generated:

```text
runs/demo/
  data/manifest.csv                    # artificial volumes + labels
  qc.json                              # shape, channels, annotation summary
  unet_2d/model.pt                     # 2D trained weights
  unet_2d/evaluation/metrics.csv
  unet_2d/preview.png
  unet_3d/model.pt                     # 3D trained weights
  unet_3d/evaluation/metrics.csv
  unet_3d/preview.png
  spatial/cells.csv                    # measured cell-level proteins and physical morphology
  spatial/neighbours_3d.csv
  spatial/proximity_2d_vs_3d.csv
  spatial/3d_cell_atlas.png
  fusion/exploratory_clusters.csv     # morphology-only, protein-only, combined
  demo_summary.json
```

**Do not present demo Dice scores or marker clusters as results from human tissue.** Synthetic samples are simplistic ellipsoids with artificially assigned channels; the split is technically disjoint but has only one held-out simulated patient.

## Commands for real-data workflow

1. Download and examine the **Kuett et al. 3D imaging mass cytometry** data from <https://doi.org/10.5281/zenodo.4752030>. Study: <https://doi.org/10.1038/s43018-021-00301-w>; original analysis: <https://github.com/BodenmillerGroup/3D_IMC_publication>.
2. Inspect actual TIFF dimensions/axes in Fiji or with `tifffile.TiffFile(...)`, check marker panel and z-position metadata. **Do not assume that TIFF order is CZYX or ZCYX**.
3. Choose an already-registered 3D volume and a matching **integer-labelled instance segmentation** volume (0=background, 1..N=cell IDs). Confirm whether its masks are manual, automated or silver-standard and whether reference instances span Z appropriately. A semantic-only tissue mask is *not* an instance mask.
4. Record actual physical spacing in micrometers (Z,Y,X), and verify the channel names match the volume's exact channel order. Map a suitably nuclear channel to `Nuclear` if warranted by actual marker metadata. If there is no valid instance mask, start with descriptive analysis rather than supervised segmentation.
5. Convert using explicitly verified axes. **The following paths/axes/names are placeholders, not confirmed properties of the Zenodo files**:

```bash
python -m tissuemap.cli convert \
  --image /path/to/registered_3d_image.tiff \
  --mask /path/to/verified_instance_mask.tiff \
  --image-axes CZYX --mask-axes ZYX \
  --channels Nuclear,CD8,panCK,CD31 \
  --sample-id specimen_01 --patient-id patient_01 --split train \
  --spacing 2.0 1.0 1.0 \
  --output data/real/specimen_01.npz \
  --manifest data/real/manifest.csv
python -m tissuemap.cli inspect --manifest data/real/manifest.csv --out runs/real_qc.json
```

For multiple volumes, append one row per specimen via `convert`. Splits are assigned **by patient before training**; the manifest loader rejects a single patient's data being assigned to different splits. The originally published HER2-positive models may come from the same patient; if so they do not form independent train/test patient cohorts. Do not report cross-patient generalization based on them.

```bash
python -m tissuemap.cli train --manifest data/real/manifest.csv --model 3d \
  --channel Nuclear --epochs 25 --patch 12 48 48 --steps-per-sample 8 \
  --output runs/real3d
python -m tissuemap.cli train --manifest data/real/manifest.csv --model 2d \
  --channel Nuclear --epochs 25 --patch 12 48 48 --steps-per-sample 8 \
  --output runs/real2d
python -m tissuemap.cli predict --manifest data/real/manifest.csv \
  --checkpoint runs/real3d/model.pt --split test --patch 12 48 48 \
  --output runs/real3d/predictions
python -m tissuemap.cli evaluate --manifest data/real/manifest.csv \
  --predictions runs/real3d/predictions --output runs/real3d/evaluation
python -m tissuemap.cli quantify --manifest data/real/manifest.csv \
  --split test --output runs/real_spatial
python -m tissuemap.cli fuse --cells runs/real_spatial/cells.csv \
  --output runs/real_fusion --clusters 2
```

### Scientific boundaries

- **Supervision:** 2D/3D U-Net uses the *binary foreground* derived from reference instance masks. The metric reported is foreground Dice/IoU/HD95, **not cell instance-segmentation accuracy**. Predicted connected components are an illustrative approximation; connected touching cells require watershed, boundary learning or instance segmentation validation.
- **Input:** a single explicitly selected `Nuclear` image channel is the training input for both 2D and 3D models. Molecule channels are held for downstream independent description to reduce circular interpretation.
- **Fusion:** single-cell shape and multiplexed protein features are scaled and combined with *unsupervised clustering*, compared against morphology-only and protein-only clusterings. This is a **minimal multimodal integration experiment**, not a learned disease predictor.
- **Cell identity:** per-sample 75th-percentile CD8/panCK/CD31 intensity is only an *exploratory marker-high gate*, never validated CD8 T cell, tumor-cell, or vascular annotation. Independent gating and histology checks are necessary for biological claims.
- **2D vs 3D spatial:** nearest same-Z-section distance versus full 3D distance is explicitly defined. Cells with no same-Z target have NaN 2D distance; do not simply project all 3D cells into the 2D plane.
- **Validation:** the default synthetic example has one test sample, no biological diversity, no batch effects and no performance generalizability. Real-data studies must use independent patients and external datasets where possible, record confidence intervals and annotation agreement.
- **Imaging physics:** 3D IMC from aligned serial sections is not the same as optical Z-stack confocal imaging. Slice gaps/missing sections, registration errors, tissue distortions and anisotropy must be inspected.
- **Large volumes:** training is patch-based; inference is overlapping sliding-window. The converter currently materializes TIFFs in memory; for very large stacks use spatial chunking/Zarr before scaling up. The project does not currently register serial raw sections or convert MCD files to TIFF.

## Three-day skill sprint (Beijing time)

| Day | Tasks | Acceptance evidence |
|---|---|---|
| **Oct 17 — Data & 3D processing** | Install; inspect public data catalog; verify axes, marker panel, tissue registration, instance mask quality, patient IDs and voxel spacing. Run synthetic test. | `pytest -q`, synthetic demo, `qc.json`, real-data availability/provenance log. |
| **Oct 18 — 2D/3D ML** | Convert ONE fully verified 3D IMC region (if feasible). Run actual 2D and 3D training with consistent input channel and comparable splits, inspect foreground overlays and failures. | `model.pt`, `metrics.csv`, failure cases, GPU/runtime log; be explicit if only synthetic. |
| **Oct 19 — Morphology/protein/reproducibility** | Calculate physical cell volumes/elongation, candidate marker intensities, 3D cell–cell edges and nearest neighbors, joint morpho-protein clusters; write concise interpretation and limitations; push to GitHub if desired. | `cells.csv`, `neighbours_3d.csv`, `exploratory_clusters.csv`, `3d_cell_atlas.png`, documented README and tests. |

## How to describe the project in an Oxford job application (only after real-data work is done)

"Developed a reproducible prototype integrating 3D biomedical image segmentation, spatially grounded tissue morphology, and multiplexed protein measurements. Implemented comparative 2D/3D U-Net baselines, physical-space cell neighbourhood analysis, and reproducible command-line evaluation." **Do not claim comparative improvement or validated clinical phenotypes without verified real-data results.**

## Sources and attribution

- Kuett L, Catena R, Özcan A, et al. *Three-dimensional imaging mass cytometry for highly multiplexed molecular and cellular mapping of tissues and the tumor microenvironment.* Nature Cancer 3, 122–133 (2022). DOI: <https://doi.org/10.1038/s43018-021-00301-w>.
- Original data: <https://doi.org/10.5281/zenodo.4752030>.
- Original preprocessing and cell analysis code: <https://github.com/BodenmillerGroup/3D_IMC_publication>. **This repo is a new teaching/portfolio prototype and does not reproduce the authors' published performance.**

## Scope of this release

This first release deliberately avoids a huge framework, clinical assumptions, a 3D foundation model, or invented molecular labels. The next technical upgrade is instance-aware segmentation plus cross-patient validation, once the actual reference masks and patient-level sample inventory have been verified.
