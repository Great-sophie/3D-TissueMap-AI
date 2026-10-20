# Manuscript main-figure draft legends

These captions are **drafts based on the frozen CSV results**. Numeric findings must be checked against the rendered panels. All confidence intervals are within-volume spatial-block resampling intervals and do not establish between-patient generalization.

## Figure 1. Study design and 3D IMC source measurements
(A) Study design for fixed three-dimensional pseudo-instances, leave-one-middle-Z signal prediction and source-only risk model transfer. (B) SecondHer2 composite: Ir193 nuclear channel in blue, Gd156 signal in magenta, source pseudoinstance boundaries in yellow, selected section. (C) LVILymph composite with corresponding isotope channels. Channels are dataset-specific; Gd156 is provisionally assigned c-erbB- in LVILymph; **antigen and sample provenance must be confirmed**. Panels (B-C) need visual registration QC before submission. Target protein signal is held out, but object geometry is derived from the existing full-volume segmentation.

## Figure 2. Z-distance and held-out measurement error in SecondHer2
(A) Median absolute log1p single-plane prediction error versus Z lag for HER2 3B5 and HER2 29D8, annotated with numbers of eligible pseudo-instances. (B) Median within-object difference relative to lag 1, restricting to matched objects at each lag. Numbers decrease at longer lags; the curves do not define a universal tissue correlation length.

## Figure 3. Cross-volume evaluation of Z sampling policies
(A-B) Held-out middle-Z median absolute log1p error for Nearest-1, Nearest-2 and all remaining sections in SecondHer2 and LVILymph respectively. (C-D) Paired **mean** error reduction for Nearest-2 relative to Nearest-1, with spatial-block bootstrap 95% quantile intervals. Median bars and mean comparisons are different estimands. Independent volumes are not proven independent patients.

## Figure 4. Portable measurement risk model transfer
(A) Three-fold spatial out-of-fold AUROC in SecondHer2 source pseudo-instances. (B) Frozen source HER2 3B5 model evaluated on LVILymph Gd156 for predicting upper-quartile held-out Z-plane measurement error. (C) Exploratory cross-marker transfer of the source-trained model to five protein channels in one LVILymph volume. Evaluation labels use each target marker’s within-volume q75; no LVILymph labels are used for model fitting.

## Figure 5. Feature ablation and cross-volume limitations
(A) Comparison of variation baseline, geometry-only, signal-only and combined models with/without explicit log absolute intensity. (B) Paired block-bootstrap AUROC contrasts in the primary cross-volume mapping. (C) Conditional selective error reduction when retaining the lowest-risk half of pseudo-instances by marker. (D) Feature shift between source and target distributions; a major shift in absolute intensity limits unqualified transfer claims.

## Submission gate (must review before journal submission)
- Confirm LVILymph Gd156 antigen identity against full antibody panel, not the truncated `c-erbB-` label.
- Confirm physical Z spacing, registration, 16-of-17 section selection and biological sample independence.
- Inspect both Figure 1 images at full resolution; matched image-mask dimensions alone are insufficient.
- Cite original 3D IMC acquisition and segmentation methods; source IDs are *pseudo-instances*, not independent manual cell ground truth.
- Distinguish within-target q75 proxy risk AUROC from fixed-source absolute error thresholds.
- Treat marker-level outcomes from one volume as correlated; do not imply five patient-level replications.
- Report unsuccessful hotspot filtering and geometric ablations transparently in supplementary results.
