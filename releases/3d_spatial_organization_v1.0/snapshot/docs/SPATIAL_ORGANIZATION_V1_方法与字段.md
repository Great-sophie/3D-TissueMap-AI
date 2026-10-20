# Workflow 08 v1.0 — 方法学与字段定义

## 输入与来源

1. 已知的作者自动 3D watershed 实例 TIFF；标签 `0` 为背景，正整数为作者实例 ID。
2. Workflow07 的 `cell_morphometry.csv`、`cell_slice_profiles.csv`、`cell_morphology_qc.csv` 和带 SHA-256 的 `morphometry_summary.json`。
3. 严格校验 mask 文件 SHA-256、ZYX 形状、实例 ID 集合、实例体素总数、每个 Z 层的 `(ID, area)`。未证实 TIFF 的 `QYX` 等元数据轴字母能表示物理 Z 顺序。

## 图 1：共享 Z 的 XY 半径邻域（`spatial_neighbors.csv`）

逐个整数 Z 切片取 Workflow07 各作者实例在该切片的二维质心 `(centroid_y_px, centroid_x_px)`；通过 cKDTree 找到该平面内质心欧氏距离 `d_xy <= R` 的不同 ID 对。跨切片使用同一组实例 ID，重复无向边合并；记录满足阈值层里的最小距离 `min_xy_plane_centroid_distance_px`、该层 `min_distance_z_index` 及 `shared_z_planes_within_radius`（不是两标签的总共享层数）。

对每个节点，在**R 内候选**里按照 (距离, ID) 排名，得到 `rank_a_within_radius` / `rank_b_within_radius`；`topk_by_either` 与 `topk_by_both` 指两端是否满足 k 以内。超过半径的节点不参与全局 kNN，也不做向外搜邻居。一个实例跨多个 Z 层，但邻接对仍是唯一无向边。

## 图 2：网格体素面界面（`instance_contact_graph.csv`）

对于 `(z,y,x)` 面上的 6-邻接方向，若两个体素都大于 0，且标签不同，生成有序实例对 `(minID,maxID)` 并将对应方向计数加 1。每个内部相邻体素面只数一次：

- `z_face_count_voxel_grid`: 相邻 Z 切片的匹配 XY 位置、但标签不同。
- `y_face_count_voxel_grid`: 同一切片内沿 Y 相邻。
- `x_face_count_voxel_grid`: 同一切片内沿 X 相邻。
- `xy_face_count_voxel_grid = y + x`。
- `all_face_count_voxel_grid = z + y + x`。

Workflow07 的 `other_instance_interface_faces_voxel` 是**每个实例的定向边界面数**，因此总和应等于此处界面面数的 2 倍；执行时自动核查。

## 图 3：逐实例空间特征（`instance_spatial_features.csv`）

- `n_xy_radius_neighbors`: 共享 Z 的 XY 半径图无向邻居数。
- `n_xy_radius_limited_topk_either`、`n_xy_radius_limited_topk_mutual`: 半径内满足任一端或双端 top-k 的边数。
- `nearest_xy_within_radius_px`: 半径内最近邻的层内 XY 质心距离，孤立节点为缺失值，**不等于完整空间最近邻距离**。
- `n_xy_face_interface_neighbors` / `n_z_face_interface_neighbors`: 网格面接触图中对应方向的不同实例邻居数量。
- `total_neighbor_interface_faces_voxel_grid`: 与其他正整数标签共享的网格面数量。
- `qc_flag_review`, `touches_z_stack_boundary`: 继承 Workflow07 的启发式标记；绝不可直接解释为真/假细胞。

## 论文/面试用正确表述

> We constructed two distinct, provenance-verified spatial representations from a single author-segmented 3D imaging mass cytometry specimen: an in-plane centroid proximity graph restricted to co-occurring Z sections, and a voxel-face adjacency graph that retains axis-specific interface counts. All measurements remain in image-index units because physical Z spacing and biological instance identity have not been independently validated.

## 已知限制

这一版**不是**细胞表型网络、细胞类型空间富集、蛋白-形态联合分析、3D 各向同性邻域、细胞膜接触或多病人统计检验。不要将图模型的静态拓扑视为确认的生物学相互作用网络。后续应先验证多通道配准、Z spacing 与蛋白实例表的映射，再延伸到空间蛋白网络。
