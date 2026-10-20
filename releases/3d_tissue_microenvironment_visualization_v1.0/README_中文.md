# Workflow 10 — 3D Tissue Microenvironment Visualization v1.0

**状态：已冻结的探索性、离线可视化基线（不代表生物学验证通过）。**

## 本 release 包含

- `viewer/centroid_marker_viewer.html`：12,663 个作者 watershed 伪实例质心、五通道信号与 QC 交互选择。
- `viewer/roi_voxel_viewer.html`：前景最密集 192 × 192 px ROI 的抽样掩码体素交互图。
- `viewer/viewer_summary.json` + `viewer/viewer_sha256.json`：原始输出信息及哈希。
- `hotspot_qc/her2_hotspot_blocks_120px.csv`：120 px 分块的 HER2 Top 1% 描述性统计。
- `hotspot_qc/her2_core_roi_registered_overlay.png`：X=480:720, Y=480:720, Z=0/5/10/15 的 HER2、panCK、Ir193 叠加核查图。
- `snapshot/`：生成 Viewer 的代码与冻结/测试脚本快照。
- `provenance_lock.json` + `release_sha256.json`：输入关联、SHA-256 和版本完整性。

## 可复现性与局限

- Viewer HTML 的生成代码已存档；Viewer 输入文件仍保存在原项目 `runs/`、`data/` 中。
- HER2 ROI 叠加图与热点 CSV **原先通过交互式单次命令生成，未保存为可独立执行的工作流源码**。本次归档了它们的原件及哈希，不能宣称它们已完全代码级复现。
- XY 为像素坐标，Z 仅为 0–15 切片索引；未验证物理 Z spacing。
- 仅一个标本；伪实例不等于真实完整细胞身份；膜相关蛋白可能不在核样 mask 内。
- 高信号形态与同域出现有视觉支持，不代表 HER2 IHC 评分、细胞阳性表型、精准配准或跨标本推断。
- `viewer/` 内 HTML 为包含 Plotly 运行库的离线文件；可直接在浏览器中打开，无须网络。

## 验证

在项目根目录：`python tools/freeze_workflow10.py --verify`。
验证同时检查 Release 本身和其绑定的现有输入、原始 QC 文件及 Workflow09 release。
离线查看可直接打开 `releases/3d_tissue_microenvironment_visualization_v1.0/viewer/centroid_marker_viewer.html`。
