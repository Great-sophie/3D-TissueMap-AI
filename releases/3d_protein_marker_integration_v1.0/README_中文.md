# Workflow 09 — 3D Protein Marker Integration v1.0

正式状态：**exploratory mask-restricted signal and spatial association baseline**。本次归档来自真实的 Workflow09C/09D 表格及代码 SHA-256 核验。

- 正式脚本仍位于 `scripts/`，归档副本仅用于复现与审计。
- 输出仍位于 `runs/3d_protein_marker_integration_v1/` 和 `runs/3d_protein_spatial_association_v1/`。
- 五通道：Ir193 / panCK / HER2 / CD31 / LYVE1；12,663 个作者三维 watershed 伪实例。
- `XY` 使用像素坐标；`Z` 是切片索引。未验证真实微米间距或独立配准精度。
- 核样掩码中的强度不能定义为完整细胞膜/胞质蛋白表达；不能据此判定阳性/阴性。
- 作者强度表 6,455 行与 mask ID 的逐行顺序没有得到证实；未进行错误合并。
- 当前关联仅为**单标本探索性描述**，不能进行病人层面推断。

验证方式：`python tools/freeze_workflow09.py --verify`。
