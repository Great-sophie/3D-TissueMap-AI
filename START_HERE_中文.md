# 3D-TissueMap AI｜三天实战指南

**目标：** 为牛津 Biomedical Imaging / Histopathology AI 博后申请准备一个端到端、可验证的技术作品。不是三天完成新论文。

## 已经完成（代码原型）

- 模块 1：真实可训练的 2D / 3D U-Net（PyTorch），支持 3D Patch sampling 和 3D sliding-window 推理。
- 模块 2：标记图像轴顺序、voxel spacing、3D segmentation mask 的数据 QC；物理单位下的细胞体积、伸长度、坐标与邻域。
- 模块 3：同一三维位置的细胞形态和多通道蛋白强度整合；形态-only、蛋白-only 和融合聚类。
- 模块 4：患者级分割防泄漏、Dice、IoU、HD95 和可视化 QC。
- 模块 5：CLI、pytest、GitHub Actions、README、配置、可复现合成数据。

自动化测试已通过 8/8 项。**用人工模拟的数据验证了代码通路。尚未运行真实 Kuett 3D IMC 数据，不能称作真实病例分割结果。**

## 第一件事：在你的 Ubuntu 环境运行

已有 `tempodino` conda 环境安装 torch 时：

```bash
unzip 3D-TissueMap-AI.zip
cd 3D-TissueMap-AI
conda activate tempodino
pip install -e '.[dev]'
pytest -q
OMP_NUM_THREADS=2 python -m tissuemap.cli demo --output runs/demo --epochs 10
```

如希望隔离依赖，可创建新的 Python 3.11 环境后安装。首次安装可能需要下载 PyTorch 包。检查 `python -c "import torch; print(torch.cuda.is_available())"` 是否识别到 CUDA。

## 10 月 17 日：数据与空间信息

1. 运行以上命令，查看 `runs/demo/unet_3d/preview.png`、`runs/demo/spatial/3d_cell_atlas.png`。
2. 打开 <https://doi.org/10.5281/zenodo.4752030>，确认压缩文件及每个 TIFF 的实际轴和通道定义。
3. 手工选取一个已经严格配准的 3D 图像和相匹配的参考 instance mask；如无可用 mask，暂不做真实数据监督训练。
4. 查到 **真实** voxel spacing、patient ID 与标注来源后，再使用 `convert` 命令；参考 README 的输入示例。不能用示例的间距替代真实值。

**验收：** 真实数据记录表 + 成功的 QC（可能由于 Zenodo 下载/标注问题而延后）。

## 10 月 18 日：模型训练与比较

1. `train --model 2d` 和 `train --model 3d`；两者用相同的 Nuclear 通道和分割标签。
2. `predict` 输出概率与二值分割；`evaluate` 保存 Dice/IoU/HD95。
3. 检查 segmentation overlay 和失败样本。**若只有一个患者，禁止写“跨患者泛化”，若没有真实标签则只能用合成数据做训练演示。**

**验收：** `model.pt`、`metrics.csv`、2D/3D 图像、实验日志。

## 10 月 19 日：形态学 + 分子信息 + 软件整理

1. `quantify` 得到 `cells.csv`、三维邻域和 2D/3D 近邻距离。
2. `fuse` 得到 `exploratory_clusters.csv`。这是探索性联合聚类，不是已验证的肿瘤/免疫亚型预测。
3. 在 GitHub README 写出真实完成的工作、局限、重现命令和图例。
4. 挑选 1 张 3D 图、1 张 segmentation overlay、1 张评价表准备面试。

**验收：** 一个清楚可复现的研究作品，外加两段不夸大的技术描述。

## 需要你提供的最少信息

把 **Zenodo 数据下载后实际看到的文件名/目录结构**（`find` 或 `ls` 输出）贴给我。我就能针对真实格式写专用数据转换脚本，不猜测 `.tiff` 轴顺序和 mask 对齐方式。

## 科研解释的红线

- 合成图像的测试 Dice 只证明系统能运行，不是病理学性能。
- 当中的候选 CD8-high / panCK-high 标签只是强度门控，不能当作 T 细胞或肿瘤细胞的真值。
- 3D IMC 是连续组织切片重建，不等同光学 Z-stack。
- 只有真实独立患者和高质量参考标注才能支撑可泛化的模型性能比较。
