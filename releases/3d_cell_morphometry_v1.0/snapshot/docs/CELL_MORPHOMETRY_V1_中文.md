# Workflow 07 — 3D Cell Morphometry v1.0

## 研究定位

输入是作者的自动 3D watershed **pseudo-instance** 标签图（通常名为 `full_mask*.tif`），不是专家验证的单细胞真实标注。该模块**不再训练、修复、重标注或调整**已冻结的 Instance Segmentation v1.0。所有结果均是对输入实例标签的描述性统计，不能被表述为真实细胞测量精度或样本外生物学结果。

作者原始 IMC 数据的 Z 层顺序/配准、物理 Z 间距和蛋白表映射尚未独立核实。本模块只根据**用户显式确认的 Z,Y,X 存储顺序假设**计算 Z 索引、XY 像素坐标和 voxel-based 量。不能换算为 µm、µm²、µm³，也不能将 `surface_faces_per_voxel_to_two_thirds` 称为物理球形度。

## 安装（不覆盖冻结代码）

把 `3D-TissueMap-Cell-Morphometry-v1-Update.zip` 放在项目父目录。Ubuntu 运行：

```bash
cd "${PROJECT_ROOT}"
conda activate tempodino
unzip -n "../3D-TissueMap-Cell-Morphometry-v1-Update.zip" -d .
python -m pytest -q tests/test_cell_morphometry_v1.py
```

`unzip -n` 遇到已有同名文件不会覆盖。新增文件：
- `scripts/cell_morphometry_core.py`
- `scripts/run_3d_cell_morphometry.py`
- `tests/test_cell_morphometry_v1.py`
- `docs/CELL_MORPHOMETRY_V1_中文.md`

依赖：numpy, scipy, pandas, tifffile, matplotlib, pytest（既有 `tempodino` 环境预计已装；如果缺失，用环境包管理器安装）。无需 PyTorch / CUDA / GPU。

## 运行

先检查真正的原始 mask 名称：

```bash
ls -lh data/inspect/LVILymph/full_mask*.tif*
```

**必须确认文件确实是按 Z,Y,X 存储的 3D 实例 ID（且背景=0）**。之前的 TIFF axes 字段可能是 QYX，不能把 Z 当作已获物理验证，仅能使用项目审计的 slice-index 对应关系作为工作假设。完成这个检查后显式加 `--assume-zyx`：

```bash
python scripts/run_3d_cell_morphometry.py \
  --data-dir data/inspect/LVILymph \
  --out runs/3d_cell_morphometry_v1 \
  --assume-zyx
```

也可以指定确定的原始作者 mask，注意在引号内写路径：

```bash
python scripts/run_3d_cell_morphometry.py \
  --mask "data/inspect/LVILymph/full_mask_final_segmentation_hwatershed_bg500_90%.tif" \
  --out runs/3d_cell_morphometry_v1 \
  --assume-zyx
```

**同一路径已有输出则直接拒绝运行，不会覆盖实验。** 要开始新实验，请使用不同 `--out`。可选参数 `--min-qc-voxels 12` 指定“小实例”审查启发式阈值（不是生物学质量标准）。`--no-figures` 跳过 PNG。由于这是单个标本上的伪实例，所有图都只供方法审计。

## 输出与字段

`runs/3d_cell_morphometry_v1/`：

| 文件 | 内容 |
| --- | --- |
| `cell_morphometry.csv` | 每个源 ID 一行；体素数量、XYZ index/pixel 质心、Bounding box、Z-span、逐层面积摘要、界面 faces |
| `cell_shape_features.csv` | 同一 ID 的形态特征子集，含连续 XY 截面质心最大步长与 voxel-face 代理指标 |
| `cell_slice_profiles.csv` | 每个 `(instance_id,z_index)` 非空组合一行；XY 截面积（像素数）、该层 XY 质心 |
| `cell_morphology_qc.csv` | 6/26 邻接连通分量数、最大 6CC 占比、上下/XY 边界接触、审阅原因 |
| `morphometry_summary.json` | 标本汇总、环境版本、作者 mask 路径与 SHA-256、轴假设、解释限制 |
| `output_file_sha256.json` | 每个输出 CSV/JSON/PNG（除自身）的 SHA-256 |
| `figures/` | voxel-count 分布、Z-span、QC flags、示例 Z-area profile、3D voxel-index 点云（仅演示，非物理尺度） |

解释各关键字段：

- `voxel_count`：标签 ID 的体素总数，**不是物理体积**。
- `centroid_z_index`：数组层索引；`centroid_y_px`、`centroid_x_px`：XY 像素坐标，0-based。
- `z_span_planes`：`z_max_index-z_min_index+1`；`n_slices_present`：实例真正出现的层数。
- `z_gap_planes`：在首末层之间未出现该 ID 的层数；大于 0 表示跨层断档，并不一定代表真实细胞断裂。
- `bbox_occupancy_fraction`：体素数除以该 ID 的 3D 最小包围盒体素数。因 XY/Z 比例未知，不能当作物理细胞密度。
- `exposed_*_faces_voxel`：每个体素邻接面上遇到背景或**另一实例 ID**的面数；两种实例交界每个 ID 各计一个面。Z 和 XY 的面面积物理意义不同，**不能将这些面计数视为校准的物理表面积**。
- `other_instance_interface_faces_voxel`：与其他 ID 的 voxel-face 界面总数，不代表真实细胞接触。
- `components_6connected` / `components_26connected`：直接依标签图计算的 3D 连通分量数，反映伪标签的连通性，而非独立证实的细胞身份。
- `qc_flag_review`：由 **small size / gap / multi-CC / FOV border** 规则触发的人工复核提示；**不能当作分割错误率**。

## 运行后的核验命令

```bash
python - <<'PY'
import json
from pathlib import Path
import pandas as pd
root=Path('runs/3d_cell_morphometry_v1')
s=json.loads((root/'morphometry_summary.json').read_text())
a=pd.read_csv(root/'cell_morphometry.csv')
p=pd.read_csv(root/'cell_slice_profiles.csv')
q=pd.read_csv(root/'cell_morphology_qc.csv')
assert len(a)==len(q)==s['n_instance_ids']
assert a.voxel_count.sum()==p.area_px2.sum()==s['foreground_voxels']
assert a.instance_id.is_unique and q.instance_id.is_unique
print({k:s[k] for k in ('shape_assumed_zyx','n_instance_ids','foreground_voxels','n_multiplane_ids','n_z_gapped_ids','n_multicomponent_6','n_qc_review_flagged')})
print('PASS: ID uniqueness, per-slice area/voxel total consistency.')
PY
```

## 接下来的研究决策

1. 检查伪实例的 voxel-count、Z-span 与连通性分布，筛选人工 QC 样本；不要为了美化结果更改源 mask。
2. 先核实作者形态学 / 蛋白表的细胞级 ID 对应关系，确认有相同的空间与 label-ID 语义，再考虑表格联结。
3. 取得可信 Z spacing 及 XY 标定，核实方向/配准之后，另建版本计算物理形态；不得在 v1 数据表里悄然添加物理单位。
4. 后续 Spatial Organization 所有发现均标明来源为 **single-specimen automated pseudo-instances**。
