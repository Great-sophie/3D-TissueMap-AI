# Reproducibility and data policy

## Local prerequisites

The project uses Python, NumPy, pandas, tifffile, scikit-learn, SciPy and
Matplotlib in different stages; individual scripts may require extra packages.
Dependencies have **not** been pinned from the creator's live environment.
Install them in a dedicated environment, and inspect imports for each entry point.

## What is excluded from GitHub

Do not upload `data/`, `runs/`, full TIFF/OME-TIFF stacks, raw segmentation masks,
biological sample metadata, credential files, Jupyter checkpoints, or local
ZIP downloads. Raw inputs must be acquired from their original provider
subject to licensing and patient privacy requirements.

## Run provenance

`manifests/code_sha256.csv` is an immutable snapshot of exported code hashes.
The original large `runs/` outputs remain on the local system and should be
backed up separately. Figures derived from data must be checked against frozen
output tables before publication.

## Interpretation limits

- Predicted heldout sections were not included as *signal* inputs, but the
  3D pseudoinstance mask is based on the complete volume.
- Source-versus-target transfer may experience significant intensity shift.
- Different markers and volumes are not independent biological replicates.
- Signal-only models and simple baselines can outperform geometry-augmented
  models; negative findings should be retained.
