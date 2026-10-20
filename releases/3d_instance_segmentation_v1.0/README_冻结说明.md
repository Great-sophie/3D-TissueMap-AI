# 3D Instance Segmentation v1.0｜正式冻结说明

冻结版本：`3d_instance_segmentation_v1.0`；生成时间（UTC）：2026-10-17T17:07:16.567815+00:00。

## 研究状态（必须保留）

- **已完成的工程阶段**：三维语义分割、Watershed、Voxel Affinity、Supervoxel、Static/Dynamic/Learned RAG、Z-Continuity Rescue、误差归因及 Z 配准/身份审计。
- **不是已验证的最优三维实例分割算法**：用户报告同一反复使用的 VAL 图块 `(384,576)` 上，Distance Watershed PQ ≈ **0.118384**；Learned Logistic RAG PQ ≈ **0.083777**；Learned RAG + Z Geometry Rescue PQ ≈ **0.099232**。Geometry Rescue 匹配实例 TP 从 404 降至 294，不能只看 PQ 增加。
- 参考均为**作者生成的 Watershed pseudo-instance labels**，而非独立人工 3D cell ground truth。
- 仅**单份 3D IMC 标本**，TRAIN 空间图块有关联；VAL 多次用于开发，不是干净的独立测试；**TEST 不得用本冻结工具访问、重评或调参**。
- Ir193 图像/掩码跨切片精确配准、物理 Z-spacing、实例 ID 与 protein CSV 的对应关系仍未完成独立核实。

## 冻结内容

- `snapshot/scripts/` 与 `snapshot/tests/`：保留原项目路径结构的代码快照；如果原项目包含后续 Morphology 脚本，其归类为“下一工作流”，不代表纳入 v1.0 的方法学结论。
- `snapshot/runs/`：仅拷贝选定实验目录中的文本结果/CSV/JSON、（按本次运行选项）图像与 checkpoint；**不会拷贝原始数据或大型中间体数据**。
- `script_workflow_index.csv` 和 `WORKFLOW_MAP_中文.md`：按科研工作流对源脚本分类。源 `scripts/` 未搬动，以免打断 Python 导入。
- `frozen_metric_comparison.csv`：直接读取本地 VAL CSV 汇总已有 PQ，不重算、不选择最优阈值。
- `MANIFEST_SHA256.csv`：每个冻结文件的 SHA-256；用于审计来源及验证未被修改。
- `excluded_files.csv`：因体积上限而未快照的文件。缺失关键模型需解决后才能声称完整恢复环境。
- `source_run_inventory.csv`：已有实验目录与文件统计（仅项目本地）。
- `runtime_info.json` 和 `pip_freeze.txt`（如成功）：记录冻结当时的 Python 环境；不等于完整 CUDA/Conda 环境复现。

快照包含文件：**190**；总计 **81.7 MiB**；因文件大小限制跳过 **0** 个文件。

## 用法与安全

1. **继续在项目根目录运行原脚本**，不要把原始 `scripts/`、`tests/`、`runs/` 直接移动到工作流分类目录。冻结快照用于回溯、审计和存档。
2. 原始 IMC 数据仍在 `data/`，没有复制到 release。真正跨机器重跑还需要数据来源、hash、路径映射以及受检模型的环境依赖。
3. 用户应手动执行本项目关键测试，并核查 `excluded_files.csv` 和 manifest；本工具**不执行任何模型训练或独立重复结果验证**。
4. 如需校验冻结后的快照：`python tools/freeze_instance_segmentation_v1.py --verify-release`。
5. ZIP 增量包只在**单独显式**运行 `--archive-zips-only --apply` 时移入项目父目录的 `_archive/instance_segmentation_v1_updates/`，不会自动删除。

**下一独立工作流**：`3D Cell Morphometry`。继续使用 `runs/` 新路径，避免修改此冻结 release。
