# Main results and evidence boundaries

**Status:** Results from frozen local CSV files; this document is assembled for repository display and is not a journal-accepted result.

## Data and estimands

- Source: SecondHer2BreastCancerModel, 92 sections, 2 HER2-labelled signal channels.
- Target: LVILymph, 16 aligned sections and 5 reported marker channels in the external evaluation.
- Objects are source-mask **pseudo-instances**. Source/target held-out mean signals are observational proxy references, not independent ground-truth single-cell concentrations.
- All comparisons are within a known segmented 3D mask. The withheld plane is a signal holdout, not a missing-plane cell reconstruction benchmark.
- Spatial-block confidence intervals quantify conditional within-volume variation, not between-patient replication.

## Z-sampling benchmark

Primary measure: median per-object absolute log1p held-out-Z signal error (lower is better).

### SecondHer2

| Marker | Eligible objects | Nearest-1 | Nearest-2 | All other planes |
|---|---|---|---|---|
| HER2 3B5 (SecondHer2) | 2914 | 0.16015 | 0.13390 | 0.14703 |
| HER2 29D8 (SecondHer2) | 2914 | 0.07116 | 0.06401 | 0.06715 |

### LVILymph

| Marker | Eligible objects | Nearest-1 | Nearest-2 | All other planes |
|---|---|---|---|---|
| Gd156 / c-erbB-? (LVILymph) | 1780 | 0.09037 | 0.06507 | 0.06331 |
| panCK (LVILymph) | 1780 | 0.07608 | 0.05657 | 0.05687 |
| CD31 (LVILymph) | 1780 | 0.08486 | 0.06380 | 0.06468 |
| E-cadherin (LVILymph) | 1780 | 0.19622 | 0.13434 | 0.12798 |
| LYVE-1 (LVILymph) | 1780 | 0.20628 | 0.14855 | 0.14275 |

Nearest-2 improves on Nearest-1 for the tested channels; all-other-plane averaging can perform similarly or better for some markers. Do **not** claim nearest-2 universally outperforms all others.

## Cross-volume marker-specific risk prediction

Primary mapping: source HER2_3B5 to target Gd156; evaluation labels are the upper quartile of target within-volume Nearest-2 absolute log1p error. No LVILymph outcomes were used for model training.

| Target channel | Variation AUROC | Signal-only AUROC | Full AUROC | Full − signal |
|---|---|---|---|---|
| Gd156 / c-erbB-? (LVILymph) | 0.69673 | 0.78456 | 0.78841 | +0.00385 |
| panCK (LVILymph) | 0.70578 | 0.77190 | 0.74793 | -0.02397 |
| CD31 (LVILymph) | 0.71736 | 0.76798 | 0.74066 | -0.02732 |
| E-cadherin (LVILymph) | 0.59324 | 0.62534 | 0.63105 | +0.00571 |
| LYVE-1 (LVILymph) | 0.61062 | 0.56647 | 0.57190 | +0.00543 |

### Primary Gd156 ablation

| Model | AUROC | AUPRC | 50% low-risk mean-error reduction |
|---|---|---|---|
| variation | 0.69673 | 0.52552 | 29.72% |
| signal_only | 0.78456 | 0.61823 | 43.66% |
| full | 0.78841 | 0.60654 | 43.34% |
| geometry | 0.47411 | 0.25385 | -3.47% |
| full_no_absolute_intensity | 0.62706 | 0.40794 | 18.18% |
| signal_no_absolute_intensity | 0.73501 | 0.56326 | 36.45% |

- Full − signal-only paired ΔAUROC = +0.00385; 95% block-bootstrap quantiles [-0.01851, +0.02336].
- Geometry does not establish an independent improvement in this cross-volume analysis.
- Absolute-intensity feature shift: 95.62% of Gd156 target pseudo-instances outside the 95% source feature range (median shift 4.93 source SD).

## Negative results and limitations

- Original geometry-augmented reliability model did not consistently improve downstream hotspot selection. Hotspot-margin ranking was a stronger comparator in the internal proxy task.
- Feature importance cannot be interpreted causally as biology; intensity distribution shift is severe.
- Target channels are 5 correlated measurements on one volume, not 5 independent patient replications.
- LVILymph Gd156 `c-erbB-` identity and the physical Z spacing should be checked against original study metadata.
- Registered-image and segmentation alignment requires image-level manual review.
- Full 3D segmentation geometry is known when held-out-Z signal is predicted.

## Source data lineage

Input dataset summary CSVs (not raw data) are not included in the public repository. Their **relative paths and SHA256 checksums** appear in `manifests/results_sources_sha256.csv`. Raw stacks and per-object measurements remain in the local `runs/` and `data/` folders.

Figures in `assets/results/` are generated from the frozen summary tables. No simulated or manually invented experimental measurements are used.
