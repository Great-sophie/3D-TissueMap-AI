# Workflow 08 — 3D Spatial Organization v1.0（正式研究快照）

**状态：** Exploratory spatial graph baseline. 真实运行结果与代码版本已校验，不能等同于生物学验证。

## 日常唯一代码入口

- `scripts/run_3d_spatial_organization.py` — 主运行入口
- `scripts/spatial_organization_core.py` — XY 邻域与体素界面计算核心
- `tests/test_spatial_organization_v1.py` — Workflow 08 单元和合成测试
- `docs/SPATIAL_ORGANIZATION_V1_方法与字段.md` — 指标定义
- `runs/3d_spatial_organization_v1/` — 不可随意修改的已验收输出

`releases/3d_spatial_organization_v1.0/snapshot/` 是不可变的归档副本；**不是第二套日常执行代码**。

## 真实数据摘要

| Metric | Value |
|---|---:|
| 伪实例 | 12,663 |
| XY 半径（px） | 24.0 |
| 半径内 top-k | 5 |
| 共享 Z 层 XY 邻近对 | 118,498 |
| radius-limited kNN either / mutual | 37,544 / 23,392 |
| Grid interface pair（union） | 18,464 |
| XY / Z interface pair | 12,650 / 15,783 |
| XY / Z face count | 104,977 / 191,757 |
| Unflagged（平均邻居） | 9,215 (19.645) |
| Flagged（平均邻居） | 3,448 (16.233) |

## 解释边界

本研究只有一个标本，作者 watershed 产生的是自动实例 **pseudo-ID**。XY 距离仅在两个 ID 存在共同 Z 平面时、按该平面质心像素坐标计算；Z 尚未物理校准。Grid-face interfaces 不是已验证细胞接触；QC 差异主要可能受切片边界影响。没有验证细胞类型、蛋白到实例的映射或生物学相互作用。

## 维护命令

```bash
python tools/freeze_workflow08.py --verify
bash releases/3d_spatial_organization_v1.0/verify.sh
python -m pytest -q tests/test_spatial_organization_v1.py
```

新实验应使用不同的 `--out` 目录，**不要覆盖** `runs/3d_spatial_organization_v1/`。
