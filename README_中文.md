# 3D-TissueMap AI：Instance Segmentation v1.0 冻结后清理工具

## 重要原则

本工具**默认仅预览**。所有实际清理文件会先移动到项目内的 `_archive/instance_segmentation_v1_cleanup/<batch>/`；可以恢复。不会修改 `data/`、`runs/`、`releases/`、`.git/`，也不会移动项目的核心训练和推理入口。

清理开始前会重新计算冻结 release 中所有 manifest 文件的 SHA-256；如果任何冻结文件不完整或校验失败，将中止。

### 操作顺序

在项目根目录：

```bash
conda activate tempodino
python -m pytest -q tests/test_cleanup_instance_v1.py

# 1. 不改动任何文件：预览缓存和已冻结、未被导入的历史诊断脚本
python tools/cleanup_instance_v1.py --include-legacy-scripts

# 2. 手动确认候选后：将候选移至可恢复隔离区
python tools/cleanup_instance_v1.py --include-legacy-scripts --apply

# 3. 确认项目仍能运行
python tools/freeze_instance_segmentation_v1.py --verify-release
python -m pytest -q tests/
```

不想归档历史脚本、只清缓存与备份：

```bash
python tools/cleanup_instance_v1.py --apply
```

### 历史更新包 ZIP

父目录的 `3D-TissueMap*.zip` 不受此工具控制，使用此前已验证的冻结工具：

```bash
python tools/freeze_instance_segmentation_v1.py --archive-zips-only
python tools/freeze_instance_segmentation_v1.py --archive-zips-only --apply
```

会将旧 ZIP 归档到项目父目录 `_archive/instance_segmentation_v1_updates/`，经过 SHA-256 核验后移走原文件。其他 `.wps` / `.py` 和项目目录不受影响。

### 恢复

清理命令会打印批次 ID（例如 `20261018T012345Z_abcdef`）。要恢复：

```bash
python tools/cleanup_instance_v1.py --restore 20261018T012345Z_abcdef
```

如果某个同名文件重新创建，恢复会拒绝覆盖，请先解决路径冲突。

### 永久删除（只删除本工具生成的隔离批次）

只有确认新工作流、测试、冻结校验都成功后，才执行：

```bash
python tools/cleanup_instance_v1.py --purge 20261018T012345Z_abcdef --confirm DELETE_ARCHIVED_CLEANUP
```

**永久删除无法恢复。** 如果隔离区空间占用不大，建议直接保留。历史 ZIP 也可以保留在父目录 `_archive/`。

## 如何决定哪些脚本可整理？

`--include-legacy-scripts` 只会自动筛查符合 `audit_*.py`、`*_pilot.py`、`*_diagnostic.py`、`*_diagnostics.py`、`*_fixed_patch_check.py` 等保守命名规则的旧诊断入口。且必须同时：

- 位于根项目 `scripts/` 目录
- 在已验证的冻结快照中有**完全相同的 SHA-256**
- 未被 `scripts/`、`tests/`、`tools/` 中其他 Python 文件静态导入或直接引用
- 不属于工具的关键核心入口保护名单

静态依赖扫描不可能证明动态依赖不存在，所以即使满足条件，也**只是隔离，不是直接销毁**。非匹配脚本会保留，避免误删 `train_*.py` 或生产入口。

## 关于 runs/ 与脚本结构

`runs/` 里有大量历史实验，默认全部保留，尤其是 `segmentation_extended`、`voxel_affinity_extended`、`region_merge_predictor_baselines` 等权重、分割结果与比较 CSV。冻结脚本收录的是选定扩展名和小于大小限制的文件，因此**不应直接 rm -rf runs/**。

科研 Workflow 映射见 `releases/3d_instance_segmentation_v1.0/WORKFLOW_MAP_中文.md`。新实验从 Workflow 07 Morphometry 开始，用新目录，避免再扩充 v1 的实例分割脚本集合。
