# 3D-TissueMap AI / Workflow 07 / 3D Cell Morphometry v1.0

此 ZIP **仅添加新文件**，不覆盖 `scripts/` 中的 Instance Segmentation v1.0，也不修改冻结发布版 `releases/3d_instance_segmentation_v1.0/`、模型、实验或 `data/`。

## 安装

```bash
cd "/media/sophie/Data2T/3D-TissueMap AI/3D-TissueMap-AI"
conda activate tempodino
unzip -n "../3D-TissueMap-Cell-Morphometry-v1-Update.zip" -d .
python -m pytest -q tests/test_cell_morphometry_v1.py
```

## 运行真实作者 3D 伪实例 TIFF

首先执行 `ls data/inspect/LVILymph/full_mask*.tif*` 确认只有一个要处理的作者 mask。输入必须是整数 3D (Z,Y,X) ID 图，0 为背景。虽然曾做 Z 顺序审计，物理层间距/配准仍**未**验证；`--assume-zyx` 仅承认当前数据解释假设。

```bash
python scripts/run_3d_cell_morphometry.py \
  --data-dir data/inspect/LVILymph \
  --out runs/3d_cell_morphometry_v1 \
  --assume-zyx
```

输出：`cell_morphometry.csv`, `cell_shape_features.csv`, `cell_slice_profiles.csv`, `cell_morphology_qc.csv`, `morphometry_summary.json`, `output_file_sha256.json` 和 `figures/*.png`。

**结果限制：只报告 voxel/pixel/Z-index 单位，作者自动 watershed 伪细胞 ID，不是手工 3D GT。不能声称物理体积/表面积、跨样本泛化或真实 protein–cell 匹配。**

完整字段解释和数据 QC 命令：`docs/CELL_MORPHOMETRY_V1_中文.md`。
