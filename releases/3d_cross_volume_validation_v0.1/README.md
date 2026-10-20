# 3d_cross_volume_validation_v0.1

**科研阶段：** 跨体积验证（LVILymph）

**状态：** completed computational tests; independent biological replication unverified

**定位：** Main results: independent imaging-volume replication and marker-specific risk transfer

## 冻结的实验结论（需要用原始结果核对）

LVILymph 16 planes, 29 aligned channels; 1,780 eligible pseudo-instances; 5 tested marker channels show nearest2 > nearest1. Main target Gd156 external full AUROC 0.7884; signal-only 0.7846.

## 关键限制

Source labels Eu151 != LVILymph Eu151 (CD31). Target Gd156 c-erbB- identity provisional. Missing original 17th aligned plane, pixel size and independent patient origin require verification.

## 后续工作

Verify panel provenance, patient/case independence, physical Z spacing, masks/registration and out-of-distribution scale effects.

## 原始实验结果目录

- `runs/lvilymph_external_sampling_v0_1/`
- `runs/lvilymph_portable_reliability_transfer_v0_1/`
- `runs/lvilymph_portable_reliability_ablation_v0_1/`

## 代码快照

- `snapshots/scripts/run_lvilymph_external_validation.py`
- `snapshots/scripts/run_portable_reliability_transfer.py`
- `snapshots/scripts/run_portable_reliability_ablation.py`

## 可复现性要求

1. 原项目根目录的 `scripts/`、`runs/`、`data/` 均未移动或修改。
2. 快照脚本只用于冻结版本；如果脚本动态载入同级模块，必须在原项目根目录运行。
3. `manifest.json` 列出复制文件的 SHA256；大型原始 TIFF 和掩膜没有复制。
4. 数千个伪实例不构成数千个生物学重复，论文统计单位和外部样本独立性需要说明。
