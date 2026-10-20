# 3d_measurement_reliability_v0.1

**科研阶段：** 测量可靠性与风险预测

**状态：** manuscript_candidate; portable model is signal-driven; ablation established

**定位：** Main/Supplementary: reliability scores and source spatial OOF; no claim of independent geometry benefit

## 冻结的实验结论（需要用原始结果核对）

Frozen signal-only model transfers reasonably for some markers; full vs signal-only has no clear primary improvement. Removing absolute intensity materially changes performance.

## 关键限制

10 portable features; source-only fitting. Source/target absolute intensity distributions strongly shift, and LYVE-1 fails. Risk is heldout intensity error, not validated biological measurement truth.

## 后续工作

Report source-trained baseline, signal-only and full ablations with spatial uncertainty; freeze model and disclose OOD dependence.

## 原始实验结果目录

- `runs/secondher2_reliability_model_v0_1/`
- `runs/lvilymph_portable_reliability_transfer_v0_1/`
- `runs/lvilymph_portable_reliability_ablation_v0_1/`

## 代码快照

- `snapshots/scripts/run_secondher2_reliability.py`
- `snapshots/scripts/run_portable_reliability_transfer.py`
- `snapshots/scripts/run_portable_reliability_ablation.py`

## 可复现性要求

1. 原项目根目录的 `scripts/`、`runs/`、`data/` 均未移动或修改。
2. 快照脚本只用于冻结版本；如果脚本动态载入同级模块，必须在原项目根目录运行。
3. `manifest.json` 列出复制文件的 SHA256；大型原始 TIFF 和掩膜没有复制。
4. 数千个伪实例不构成数千个生物学重复，论文统计单位和外部样本独立性需要说明。
