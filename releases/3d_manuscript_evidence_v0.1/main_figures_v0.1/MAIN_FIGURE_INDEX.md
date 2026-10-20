# Main-figure assembly / evidence map

| Figure | Claim it can support | CSV evidence | Gate |
|---|---|---|---|
| Figure 1 | Study design and actual isotope images | Original OME and aligned TIFF, original masks | Check antigen identity and overlay registration manually |
| Figure 2 | Within-SecondHer2 Z-lag sensitivity | `z_lag_error_summary.csv` | Declining cohort size with lag |
| Figure 3 | Nearest-2 vs Nearest-1 in both volumes; all-others variable | `core_comparison_summary.csv`, `external_sampling_summary.csv`, `paired_spatial_block_intervals.csv` | Do not equate medians with paired mean differences |
| Figure 4 | Cross-volume heldout measurement risk ranking | `source_ablation_spatial_oof.csv`, `external_ablation_summary.csv` | Only one target volume; conditional proxy labels |
| Figure 5 | Geometry non-benefit, intensity dependence, marker heterogeneity | `external_paired_spatial_block_intervals.csv`, `feature_domain_shift.csv` | Strong OOD covariate shift |

**Supplementary candidates:** 3D pseudo-instance HER2 rank-discordance map and Moran’s I sensitivity; fixed-boundary method comparison; failed hotspot-risk filter; negative kernel experiments; source-mask traceability; method hyperparameters and cohort audit. Do not claim 3D universally beats genuine 2D tissue acquisition: projection-reference experiment reverses the method ranking.

All figures are provisional, generated from existing files without model retraining. No invented measurements were inserted.
