# 3D-TissueMap AI｜3D Instance Segmentation v1.0 正式冻结工具

## 为什么不直接移动项目内脚本？

当前 `scripts/` 中的 Python 文件互相使用 `from train_segmentation import ...` 一类直接导入，`runs/` 的路径也被许多命令写死。直接按主题搬动文件有较高概率让已经跑通的实验失效。本工具采用 **原路径不变 + 工作流索引 + 独立可校验快照** 的方式整理。

不会删除数据、不会清理 `runs/`、不会重训模型、不会使用 TEST 图块。默认只打印预览，只有 `--apply` 才创建文件。覆盖既存冻结版本会被拒绝。

## 安装

把 `3D-TissueMap-Instance-Segmentation-v1.0-Freezer-Update.zip` 放到 `3D-TissueMap-AI/` 的**父目录**，然后：

```bash
cd "${PROJECT_ROOT}"
conda activate tempodino
unzip -o "../3D-TissueMap-Instance-Segmentation-v1.0-Freezer-Update.zip" -d .
python -m pytest -q tests/test_freeze_instance_release.py
```

## 冻结操作

先预览：

```bash
python tools/freeze_instance_segmentation_v1.py
```

确认后**创建不可覆盖**的快照：

```bash
python tools/freeze_instance_segmentation_v1.py --apply
python tools/freeze_instance_segmentation_v1.py --verify-release
```

位置：`releases/3d_instance_segmentation_v1.0/`。

快照包含完整 `scripts/`、`tests/`、`tools/` 的受支持文本文件、既有结果目录中的 JSON/CSV/图像/模型文件（受单文件 1 GiB 限制）、环境信息、全部结果路径清单、按工作流组织的脚本索引和 SHA-256 清单。它**不**复制原始 IMC 数据、巨型 NIfTI/TIFF/NPY 中间数据；这不是完全自包含的数据分发包。超过 1 GiB 的文件会记录在 `excluded_files.csv`；可以冻结前使用 `--max-file-mb 4096` 适当调整。

## 父目录增量 ZIP：分开、显式归档

**务必确认上一步 `--verify-release` 成功之后**再执行：

```bash
# 看看将会移动哪些父目录 ZIP（不动文件）
python tools/freeze_instance_segmentation_v1.py --archive-zips-only

# 显式同意归档 ZIP：复制、核对 SHA-256、再移除旧位置的 ZIP 文件
python tools/freeze_instance_segmentation_v1.py --archive-zips-only --apply
```

目的地为同一父目录下的 `_archive/instance_segmentation_v1_updates/`。只处理父目录 `3D-TissueMap*.zip`，**不会碰** `.wps` 文件、其它 `.py` 文件、项目代码或原始数据；同名目标不会覆盖。

## Freeze 的科学结论（仅已有实验，非新增验证）

- 在反复查看过的同一 VAL 图块 `(384,576)`，Distance Watershed PQ 为 **0.1184**，Learned Logistic Region Graph PQ 为 **0.0838**，Z Geometry Rescue PQ 为 **0.0992**，其中 Geometry Rescue 将原有 404 个匹配实例降低至 294 个。
- 局部模型学习和图解码方法具有工程阶段性成果，但**仍存在严重 over-segmentation**；不能说已达到已验证的 3D 实例分割最佳性能。
- 参考实例为作者 watershed pseudo-label；仅单标本；跨层精确注册、Z 物理尺度以及独立测试尚未确认。

冻结后下一工作流：**3D Cell Morphometry / Spatial Organization**。后续新实验保持 `scripts/` 与 `runs/` 原路径，使用新的输出目录，不修改冻结 release。
