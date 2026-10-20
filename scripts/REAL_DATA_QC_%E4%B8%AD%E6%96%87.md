# 3D-TissueMap AI · 真实数据 QC（Day 1 · 下一步）

你的 LVILymph 数据已在 Ubuntu 项目内解压到 `data/inspect/LVILymph`，目录应有：
- `full_mask_final_segmentation_hwatershed_bg500_90%.tif`
- `measured_mask_final_segmentation_hwatershed_bg500_90%.tif`
- `LVILymphBreastCancer_SIMILARITY10_In115.zip`
- `LVI_lymph_labels_area.csv`、`LVI_lymph_mean_intensities.csv`、`LVI_lymph_panel.csv`
- `final_3D_stack_order.csv`

将 `qc_real3d.py` 放入 **3D-TissueMap-AI 项目根目录**，执行：

```bash
conda activate tempodino
python qc_real3d.py \
  --data-dir data/inspect/LVILymph \
  --out runs/lvilymph_real_qc \
  --slice 7
```

结果：

1. `runs/lvilymph_real_qc/real_markers_xy.png`：Ir193、panCK、LYVE1、CD31 的单切片及最大投影。
2. `runs/lvilymph_real_qc/mask_overlay_provisional.png`：第 8 层核信号和分割边界的**初步**像素叠加（不证明已配准）。
3. `runs/lvilymph_real_qc/real_qc.json`：mask 尺寸、标签个数、相邻切片的标签延续情况、蛋白强度表的行数、切片顺序表及通道文件清单。

打开结果：

```bash
xdg-open runs/lvilymph_real_qc/real_markers_xy.png
xdg-open runs/lvilymph_real_qc/mask_overlay_provisional.png
```

**把两张图和控制台最后几行输出发给我。** 重点观察掩膜边界是否贴合 Ir193 细胞核附近，以及 panCK / LYVE1 的组织位置（它们并非细胞的独立真值标注）。我们随后查证 `final_3D_stack_order.csv` 的排序与物理间距，再做细胞级形态和真正的三维距离。

科学限制：
- 掩膜 TIFF 的第一轴元数据标为 `Q`，不能仅凭 shape 判定为真实 Z 轴。
- 16 张 TIFF 的文件编号只是**候选切片顺序**，要与 `final_3D_stack_order.csv` 对照。
- `mean_intensities.csv` 没有显式 label，**严禁**未经验证直接按行与 `labels_area.csv` 合并。
- 目前不输出任何以 µm 为单位的长度或体积，不把算法生成的 watershed 掩膜当成人工标注金标准。
- 这是单个示例组织，暂时不具备患者级独立验证。
