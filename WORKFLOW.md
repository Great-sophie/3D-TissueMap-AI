# 3D-TissueMap AI — Research Workflow / 项目工作流程

唯一主项目。按科研阶段组织，不按日期分项目。

| Stage | 实验代码 | 输出目录 | 完成标准 |
|---|---|---|---|
| 01 Data Audit | `scripts/inspect_real_imc.py`, `python -m tissuemap.volume_qc` | `runs/real_volume_qc/` | axes, dimensions, panel, files, provenance |
| 02 Registration QC | `scripts/registration_qc.py`, `train_segmentation.py --audit-only` | `runs/registration_qc/`, `runs/segmentation_baselines/` | CSV ↔ TIFF filename check; mask overlay QC |
| 03 Segmentation | `scripts/train_segmentation.py` | `runs/segmentation_baselines/` | 2D/3D loss, Dice/IoU, sample predictions |
| 04 Spatial Morphology | `tissuemap.cli quantify` | `runs/spatial_morphology/` | verified-physical-spacing 3D cell metrics |
| 05 Molecular Integration | `tissuemap.cli fuse` | `runs/molecular_integration/` | verified marker-to-instance mapping |
| 06 Evaluation | tests + model metrics + provenance | `runs/evaluation/` | failure analysis and documented limitations |

## Current commands (from project root)

```bash
conda activate tempodino
pytest -q
python scripts/train_segmentation.py --data-dir data/inspect/LVILymph --out runs/segmentation_baselines --audit-only
```

检查 `runs/segmentation_baselines/data_audit.json` 的 `stack_order_check.status`，只有 `FILENAME_ORDER_MATCH` 才能继续；这不能替代真实图像/标签配准验证。

经目视复核并接受探索性对照实验的限制后执行：

```bash
python scripts/train_segmentation.py --data-dir data/inspect/LVILymph \
 --out runs/segmentation_baselines --epochs 1 --steps 10 \
 --tile 192 --patch-xy 128 --patch-z 8 --base 4 \
 --accept-provisional-registration
```

结果仅表示**同一组织区域内对自动 watershed 标签的拟合**，不是独立患者的临床泛化精度。
