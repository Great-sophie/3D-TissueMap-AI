# Workflow 10 — 3D Tissue Microenvironment Visualization v1.0 正式冻结包

本包**只新增** `tools/freeze_workflow10.py`、`tests/test_freeze_workflow10.py`，不会修改/覆盖 Workflow 07/08/09、现有 Viewer 或既有 releases。默认仅预检查，不写入 release。依赖 Python 3.10+ 标准库；运行测试需要 pytest，冻结已生成的 HTML 不需要 Plotly。

## 安装（Ubuntu）

将下载的 ZIP 放在项目的**父目录**，然后执行：

```bash
cd "${PROJECT_ROOT}"
conda activate tempodino
unzip -n "../3D-TissueMap-Workflow10-v1.0-Freeze-Update.zip" -d .
python -m pytest -q tests/test_freeze_workflow10.py
```

## 冻结（严格按顺序）

```bash
python scripts/build_3d_tissue_viewer.py --verify
python tools/freeze_workflow09.py --verify
python tools/freeze_workflow10.py
python tools/freeze_workflow10.py --apply
python tools/freeze_workflow10.py --verify
```

请确认预检查显示 `PREFLIGHT PASS: 12663 instances; 3 viewer outputs; 2 hotspot QC artifacts`，再运行 `--apply`。如果目标 release 目录已经存在，脚本会拒绝覆盖（请勿删除已成功冻结的版本）。

## 产物

```
releases/3d_tissue_microenvironment_visualization_v1.0/
├── README_中文.md
├── provenance_lock.json
├── release_sha256.json
├── snapshot/
│   ├── scripts/build_3d_tissue_viewer.py
│   ├── tools/freeze_workflow10.py
│   └── tests/...
├── viewer/
│   ├── centroid_marker_viewer.html
│   ├── roi_voxel_viewer.html
│   ├── viewer_summary.json
│   └── viewer_sha256.json
└── hotspot_qc/
    ├── her2_hotspot_blocks_120px.csv
    └── her2_core_roi_registered_overlay.png
```

两个 HTML **完整拷贝入 release**，离线打开：

```bash
xdg-open releases/3d_tissue_microenvironment_visualization_v1.0/viewer/centroid_marker_viewer.html
xdg-open releases/3d_tissue_microenvironment_visualization_v1.0/viewer/roi_voxel_viewer.html
```

本冻结记录固定来源 SHA-256，保留 Workflow 09 的 provenance，保存已审阅的 HER2 热点图。**注意：**热点分块 CSV 和三通道 ROI PNG 是先前通过交互式命令生成的，没有独立脚本可重算；本次保存的是可校验的产物，非完整研究复现流水线。单样本、伪实例、Z 未经物理标定、配准未经独立定量验证等限制已写入 release。

## 故障排查

- `Required ordinary file unavailable`: 检查项目中所需文件确实存在；本包不代替 Workflow 09/10 的前置结果。
- `Viewer SHA-256 manifest mismatch`: Viewer 与既有输出清单不一致，先停止并排查，不能直接强制冻结。
- `Workflow09 release archive changed`: 先用 `python tools/freeze_workflow09.py --verify` 核查。
- `Release already exists; refusing overwrite`: 已有版本，请直接 `--verify`，不要覆盖或清除。

**科学定位：**Exploratory 3D Pseudo-instance Viewer & HER2 Hotspot Localization, **not** cell-level phenotyping or independently verified 3D registration.
