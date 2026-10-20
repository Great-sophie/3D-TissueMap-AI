# 3D-TissueMap AI：正式科研发布目录与后续工作路线

> 这是 `releases/` 中的科研阶段归档索引，不改变现有脚本运行路径，也不代表全部任务已完成。

## 既有发布（完全保留）

| 现有目录 | 研究阶段 |
|---|---|
| `3d_instance_segmentation_v1.0` | 三维实例分割 |
| `3d_cell_morphometry_v1.0` | 细胞三维形态测量 |
| `3d_protein_marker_integration_v1.0` | 蛋白标记物整合 |
| `3d_spatial_organization_v1.0` | 空间结构组织分析 |
| `3d_spatial_protein_attribution_v0.2_exploratory` | 空间蛋白归属探索 |
| `3d_tissue_microenvironment_visualization_v1.0` | 组织微环境可视化 |

## 新增研究阶段

| 发布目录 | 核心任务 | 状态 |
|---|---|---|
| `3d_z_sampling_benchmark_v0.1` | Z 向采样与留出截面预测 | manuscript_candidate; scientific validation pending |
| `3d_spatial_hotspot_sensitivity_v0.1_exploratory` | 空间热点与失败对照（探索性） | exploratory; negative/geometry-dependent results; supplemental only |
| `3d_measurement_reliability_v0.1` | 测量可靠性与风险预测 | manuscript_candidate; portable model is signal-driven; ablation established |
| `3d_cross_volume_validation_v0.1` | 跨体积验证（LVILymph） | completed computational tests; independent biological replication unverified |
| `3d_manuscript_evidence_v0.1` | 论文图表与可复现证据归档 | evidence collection only; manuscript NOT submitted or final |

## 尚未解决的投稿前检查（必须人工核实）

1. 确认 SecondHer2 和 LVILymph 是否来自独立患者/独立采集，避免把两个体积或 1,780 个对象称作独立临床队列。
2. 核实 LVILymph 的物理 Z 间距、像素尺寸、通道来源以及第 17 张原始 OME 未进入 16 层配准体的原因。
3. 核实 LVILymph `Gd156` 原始截断标签 `c-erbB-` 的完整抗体名称；Eu151 在 LVILymph 是 CD31，不能套用 SecondHer2 通道表。
4. 论文中区分留出层强度代理误差与独立生物学真值；跨层伪实例来自完整分割体积。
5. 报告 Full vs Signal-only 的无明确几何增益、去除绝对强度的性能损失、LYVE-1 转移失败和 hotspot filtering 失败。
6. 采用 `releases/3d_manuscript_evidence_v0.1/` 内的证据快照组织候选图；输出正式 PDF/SVG，检查显微标尺、通道名称和统计单位。

## 原项目代码入口

运行正式实验仍应在项目根目录使用 `python scripts/<script>.py`，不要在 `releases/` 快照子目录直接运行。

## 防误操作与更新规则

本工具只创建新的发布目录和本索引，不覆盖或移动任何旧发布、`scripts/`、`runs/` 或原始数据。 若未来需要新版本，请复制工具并修改版本后新建目录；不要静默修改已冻结的旧版本。
