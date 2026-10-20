# 3d_spatial_hotspot_sensitivity_v0.1_exploratory

**科研阶段：** 空间热点与失败对照（探索性）

**状态：** exploratory; negative/geometry-dependent results; supplemental only

**定位：** Supplementary results: geometry-sensitive proxy hotspot reference and no successful combined-risk filtering

## 冻结的实验结论（需要用原始结果核对）

Hotspot prediction exceeds shuffle null, but XYZ-reference advantage reverses with XY-reference. Combined reliability filter underperforms hotspot margin.

## 关键限制

Reference hotspot labels are heldout-plane PROXIES using spatial neighbourhood definitions, not independent biological ground truth. Never claim 3D > 2D from reference-matched geometry.

## 后续工作

Keep as transparent exploratory/supplemental evidence; do not retune on this sample.

## 原始实验结果目录

- `runs/secondher2_3d_spatial_discordance_v0_1/`
- `runs/secondher2_3d_spatial_robustness_v0_1/`
- `runs/secondher2_3d_hotspot_recovery_v0_1/`
- `runs/secondher2_reliability_hotspot_transfer_v0_1/`

## 代码快照

- `snapshots/scripts/qc_secondher2_3d_spatial_discordance.py`
- `snapshots/scripts/qc_secondher2_spatial_robustness.py`
- `snapshots/scripts/run_secondher2_hotspot_recovery.py`
- `snapshots/scripts/run_secondher2_hotspot_transfer.py`

## 可复现性要求

1. 原项目根目录的 `scripts/`、`runs/`、`data/` 均未移动或修改。
2. 快照脚本只用于冻结版本；如果脚本动态载入同级模块，必须在原项目根目录运行。
3. `manifest.json` 列出复制文件的 SHA256；大型原始 TIFF 和掩膜没有复制。
4. 数千个伪实例不构成数千个生物学重复，论文统计单位和外部样本独立性需要说明。
