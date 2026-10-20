# 3d_z_sampling_benchmark_v0.1

**科研阶段：** Z 向采样与留出截面预测

**状态：** manuscript_candidate; scientific validation pending

**定位：** Main figures: local-Z sampling; independently report mean and median errors

## 冻结的实验结论（需要用原始结果核对）

Nearest-2 improves held-out middle-Z signal prediction over Nearest-1 in the tested SecondHer2 and LVILymph volumes. Nearest-2 is NOT universally better than all-other planes.

## 关键限制

Fixed pre-existing 3D pseudo-instance geometry includes the held-out plane; this is a signal-holdout experiment, NOT recovery of missing 3D cell geometry. Only two volumes, not verified independent patients.

## 后续工作

Check physical voxel spacing, source provenance, and produce unified cross-volume error figures with per-volume/block uncertainty.

## 原始实验结果目录

- `runs/secondher2_z_lag_profile_v0_1/`
- `runs/secondher2_nearest2_spatial_robustness_v0_1/`
- `runs/secondher2_2d_vs_3d_heldout_v0_1/`
- `runs/secondher2_core_study_v0_1/`
- `runs/lvilymph_external_sampling_v0_1/`

## 代码快照

- `snapshots/scripts/analyze_secondher2_z_lag_decay.py`
- `snapshots/scripts/evaluate_secondher2_nearest2_spatialbootstrap.py`
- `snapshots/scripts/evaluate_secondher2_2d_vs_3d_heldout.py`
- `snapshots/scripts/run_secondher2_core_study.py`
- `snapshots/scripts/run_lvilymph_external_validation.py`

## 可复现性要求

1. 原项目根目录的 `scripts/`、`runs/`、`data/` 均未移动或修改。
2. 快照脚本只用于冻结版本；如果脚本动态载入同级模块，必须在原项目根目录运行。
3. `manifest.json` 列出复制文件的 SHA256；大型原始 TIFF 和掩膜没有复制。
4. 数千个伪实例不构成数千个生物学重复，论文统计单位和外部样本独立性需要说明。
