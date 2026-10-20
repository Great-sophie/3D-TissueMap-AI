# 3D-TissueMap AI — Learned Region Graph Integration (pilot)

## 目的

将已训练的 **`logistic_balanced`** 和 **`hgb_balanced`** 区域接口分类器，接入现有 Dynamic RAG 进行**实际三维实例分割 PQ 诊断**。复用原有 900-step XYZ Affinity checkpoint，不重新训练。脚本**不会选择 TEST**。

**仅探索：** 作者 watershed 自动 pseudo-label；单一标本，VAL 已重复使用；物理 Z-spacing/注册未核实。训练在同一 TRAIN 图块上的结果为**重代入**，不能作为独立测试；VAL 也只是开发集。

## 更新文件

- `scripts/learned_region_graph.py`：加载本地可信 joblib 分类器，将 **纯 XY** 原子 Supervoxel 接口的预测 `P(same)` 转换为图合并证据；任何 Z-involved 接口均**保留原 voxel-affinity evidence**。
- `scripts/evaluate_learned_region_graph.py`：比较 Watershed、Static Greedy、原 Dynamic + Repulsion、Logistic/HGB × (score-only / original raw-repulsion veto)。
- `tests/test_learned_region_graph.py`：功能、数据泄漏屏障、Z 回退和合成三维端到端运行测试。

无需覆盖旧模块，新增的两个 Python 文件使用现有项目的 `evaluate_affinity_region_graph.py`, `dynamic_region_merging.py`, `region_interface_features.py` 等脚本。

## 运行

将 ZIP 放到项目父目录，进入项目根目录：

```bash
cd "/media/sophie/Data2T/3D-TissueMap AI/3D-TissueMap-AI"
conda activate tempodino
unzip -o "../3D-TissueMap-Learned-RAG-Integration-Update.zip" -d .
python -m pytest -q tests/test_learned_region_graph.py
```

首先只做 TRAIN smoke（**全 TRAIN 拟合分类器同数据重代入，仅供流程验证**）：

```bash
python scripts/evaluate_learned_region_graph.py \
  --data-dir data/inspect/LVILymph \
  --split-run-dir runs/segmentation_extended \
  --affinity-run-dir runs/voxel_affinity_extended \
  --predictor-dir runs/region_merge_predictor_baselines \
  --out runs/learned_region_graph_train_smoke \
  --split train --max-tiles 1 \
  --region-z 1 --region-xy 4 \
  --accept-provisional-registration
```

跑通后，可在**此前已经重复查看过的 VAL 开发图块**上做一次与旧方法对比：

```bash
python scripts/evaluate_learned_region_graph.py \
  --data-dir data/inspect/LVILymph \
  --split-run-dir runs/segmentation_extended \
  --affinity-run-dir runs/voxel_affinity_extended \
  --predictor-dir runs/region_merge_predictor_baselines \
  --out runs/learned_region_graph_val_diagnostic \
  --split val --max-tiles 1 \
  --region-z 1 --region-xy 4 \
  --accept-provisional-registration
```

输出：

- `learned_region_instance_comparison.csv`：每图块、每方法的 PQ、TP/FP/FN、实例数等。
- `learned_region_instance_comparison.json`：特征打分覆盖、合并诊断和局限性。
- `learned_region_comparison_<y>_<x>.png`：输入图像+7种方法可视化。

```bash
cat runs/learned_region_graph_val_diagnostic/learned_region_instance_comparison.csv
```

## 关键科学限制

1. **纯 XY 模型，未训练 Z。** 包含任何 Z 接触面的接口（包括 XY+Z 混合接口）都使用原始 voxel-affinity，不让 XY 模型对其进行外推。会保留已有 Z 监督不确定性。
2. **Unknown XY 接口需要推理。** 可靠伪标签只覆盖部分接口，因此分类器需要给所有纯 XY 接口打分；这是分布外/标签选择偏差的风险，脚本记录其数量但推理不使用参考 ID。
3. **动态合并不是重新预测扩大区域。** 初始 Supervoxel 接口的学习 Logit 以接触面数量为权重聚合；每次合并后基于**旧原子接口**更新图证据。没有对合并后的形态/核特征重新调用分类器，因此称为 *frozen-edge learned graph pilot*。
4. **复用旧 raw negative-share 是消融，不是 learned mutex。** `raw_repulsion` 启发式过滤与新的学习分数之间可能存在冲突。
5. **训练模型概率未校准。** 训练含类别平衡权重；默认阈值 Logit=0.2 只是预注册式固定诊断参数，不能通过这个反复使用的 VAL 挑最优阈值。预测的“概率”不应直接作为 calibrated chance 来解释。
6. 体素接触面权重不代表独立样本；这一规则是启发式聚合，不是严格概率推断。
7. **最终判断看实例 PQ、Precision/Recall、Merge/Split、实例数量**，而不是仅看接口 PR-AUC。没有独立组织和人工真值，不能宣称泛化。
8. `joblib.load` 可执行对象序列化代码，只加载自己训练产生的可信 `.joblib`。
