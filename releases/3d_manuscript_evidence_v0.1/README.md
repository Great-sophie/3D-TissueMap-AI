# 3d_manuscript_evidence_v0.1

**科研阶段：** 论文图表与可复现证据归档

**状态：** evidence collection only; manuscript NOT submitted or final

**定位：** Evidence snapshots and stage-specific result inventory for Figures 1-5, Methods and Supplementary

## 冻结的实验结论（需要用原始结果核对）

Candidate Figure 1 data and instance provenance; Figure 2 spatial structure; Figure 3 Z-sampling; Figure 4 measurement risk; Figure 5 cross-volume transfer and ablations.

## 关键限制

Candidate figures not publication-ready by default. Must check axis labels, measured-vs-pseudo-instance terminology, paper-level unit of replication, physical spacing and permissions.

## 后续工作

Select final result definitions, regenerate figures from frozen tables, prepare Methods, Results, data/code availability and target-journal formatting.

## 原始实验结果目录

- `runs/workflow12_secondher2_fixed_boundary_transfer_all92_v0_1/`
- `runs/secondher2_3d_object_protein_quantification_v0_1/`
- `runs/secondher2_3d_spatial_robustness_v0_1/`
- `runs/secondher2_core_study_v0_1/`
- `runs/secondher2_3d_hotspot_recovery_v0_1/`
- `runs/secondher2_reliability_model_v0_1/`
- `runs/secondher2_reliability_hotspot_transfer_v0_1/`
- `runs/lvilymph_external_sampling_v0_1/`
- `runs/lvilymph_portable_reliability_transfer_v0_1/`
- `runs/lvilymph_portable_reliability_ablation_v0_1/`

## 代码快照

无需新增代码快照。

## 可复现性要求

1. 原项目根目录的 `scripts/`、`runs/`、`data/` 均未移动或修改。
2. 快照脚本只用于冻结版本；如果脚本动态载入同级模块，必须在原项目根目录运行。
3. `manifest.json` 列出复制文件的 SHA256；大型原始 TIFF 和掩膜没有复制。
4. 数千个伪实例不构成数千个生物学重复，论文统计单位和外部样本独立性需要说明。
