# 脚本工作流索引（冻结时扫描生成）

这里是**索引**，不是重命名脚本或搬迁可执行路径。代码保存在 `snapshot/scripts/`。
原项目中的 `scripts/` 位置未更改，避免破坏已有导入路径和历史命令。

## 推荐复现顺序

数据 QC → 语义分割 → 实例基线 → Voxel Affinity → Region Graph 与学习合并 → 错误归因 → Z 向身份审计。
本项目实验属于单标本探索，结果不能表述为独立临床验证。

## 00_data_qc
- [`REAL_DATA_QC_%E4%B8%AD%E6%96%87.md`](snapshot/scripts/REAL_DATA_QC_%E4%B8%AD%E6%96%87.md)
- [`audit_voxel_affinities.py`](snapshot/scripts/audit_voxel_affinities.py)
- [`qc_real3d.py`](snapshot/scripts/qc_real3d.py)
- [`registration_qc.py`](snapshot/scripts/registration_qc.py)

## 01_semantic_segmentation
- [`segmentation_diagnostics.py`](snapshot/scripts/segmentation_diagnostics.py)
- [`segmentation_fit_check.py`](snapshot/scripts/segmentation_fit_check.py)
- [`train_segmentation.py`](snapshot/scripts/train_segmentation.py)

## 02_instance_baselines
- [`audit_center_boundary_heads.py`](snapshot/scripts/audit_center_boundary_heads.py)
- [`audit_instance_decoder_oracle.py`](snapshot/scripts/audit_instance_decoder_oracle.py)
- [`audit_instance_head_supervision.py`](snapshot/scripts/audit_instance_head_supervision.py)
- [`instance_linkage_qc.py`](snapshot/scripts/instance_linkage_qc.py)
- [`instance_reference_audit.py`](snapshot/scripts/instance_reference_audit.py)
- [`instance_watershed_baseline.py`](snapshot/scripts/instance_watershed_baseline.py)
- [`nucleus_guided_instance_separation.py`](snapshot/scripts/nucleus_guided_instance_separation.py)
- [`optimize_instance_separation.py`](snapshot/scripts/optimize_instance_separation.py)
- [`train_center_boundary.py`](snapshot/scripts/train_center_boundary.py)
- [`train_instance_heads_v2.py`](snapshot/scripts/train_instance_heads_v2.py)

## 03_voxel_affinity
- [`affinity_fixed_patch_check.py`](snapshot/scripts/affinity_fixed_patch_check.py)
- [`train_voxel_affinity.py`](snapshot/scripts/train_voxel_affinity.py)

## 04_region_graph_and_merge_learning
- [`audit_affinity_graph_fragmentation.py`](snapshot/scripts/audit_affinity_graph_fragmentation.py)
- [`audit_region_graph_candidates.py`](snapshot/scripts/audit_region_graph_candidates.py)
- [`audit_region_oracle_connectivity.py`](snapshot/scripts/audit_region_oracle_connectivity.py)
- [`audit_supervoxel_oracle_merging.py`](snapshot/scripts/audit_supervoxel_oracle_merging.py)
- [`audit_supervoxel_purity.py`](snapshot/scripts/audit_supervoxel_purity.py)
- [`build_region_interface_dataset.py`](snapshot/scripts/build_region_interface_dataset.py)
- [`dynamic_region_merging.py`](snapshot/scripts/dynamic_region_merging.py)
- [`evaluate_affinity_graph_pilot.py`](snapshot/scripts/evaluate_affinity_graph_pilot.py)
- [`evaluate_affinity_region_graph.py`](snapshot/scripts/evaluate_affinity_region_graph.py)
- [`evaluate_dynamic_region_graph.py`](snapshot/scripts/evaluate_dynamic_region_graph.py)
- [`evaluate_learned_region_graph.py`](snapshot/scripts/evaluate_learned_region_graph.py)
- [`learned_region_graph.py`](snapshot/scripts/learned_region_graph.py)
- [`region_interface_features.py`](snapshot/scripts/region_interface_features.py)
- [`train_region_merge_classifier.py`](snapshot/scripts/train_region_merge_classifier.py)

## 05_failure_attribution
- [`audit_foreground_graph_attribution.py`](snapshot/scripts/audit_foreground_graph_attribution.py)
- [`audit_instance_error_attribution.py`](snapshot/scripts/audit_instance_error_attribution.py)
- [`instance_error_attribution.py`](snapshot/scripts/instance_error_attribution.py)

## 06_z_identity_and_linking
- [`audit_z_link_safety.py`](snapshot/scripts/audit_z_link_safety.py)
- [`audit_z_registration_identity.py`](snapshot/scripts/audit_z_registration_identity.py)
- [`cross_slice_review_v2_core.py`](snapshot/scripts/cross_slice_review_v2_core.py)
- [`evaluate_z_continuity_rescue.py`](snapshot/scripts/evaluate_z_continuity_rescue.py)
- [`render_cross_slice_identity_v2.py`](snapshot/scripts/render_cross_slice_identity_v2.py)
- [`z_continuity_rescue.py`](snapshot/scripts/z_continuity_rescue.py)
- [`z_registration_identity_core.py`](snapshot/scripts/z_registration_identity_core.py)

## 99_shared_or_unclassified
- [`compare_inference_context.py`](snapshot/scripts/compare_inference_context.py)
- [`inspect_real_imc.py`](snapshot/scripts/inspect_real_imc.py)
- [`organize_workflow.py`](snapshot/scripts/organize_workflow.py)
- [`run_synthetic_demo.sh`](snapshot/scripts/run_synthetic_demo.sh)
