# Workflow09 方法说明：图像到三维伪实例的蛋白信号采样

## 一、输入与 provenance

本模块只依赖作者 watershed 掩码、Workflow07 的形态/QC 表和 Workflow08 的空间特征表。路径和源文件 SHA-256 通过 `marker_manifest_local.json` 和前两个 Workflow summary 验证。轴 `ZYX` 属于显式 index-grid 假设，不代表物理 Z spacing 可信。

## 二、信号聚合

对于伪实例 i 的掩码体素集合 \(V_i\) 和某个 marker m 的数值强度 \(I_m(v)\)，计算非缺失值数、均值、精确中位数、0.9 分位、标准差、总和、0 值比例。NaN/Inf 强度被排除，仍保留缺失率；有效强度 0 不被排除。没有自动背景校正、阳性阈值、对比归一化。

若 mask 主要来源于核结构，则这些统计量仅代表核区域内采样的 marker 信号。即便数据文件合法且坐标维度相同，膜/细胞质蛋白的核掩码采样不能被解释为完整细胞蛋白丰度。

## 三、二维 XY 邻域与空间表

Workflow08 的 `n_xy_radius_neighbors` 计算于共享 indexed-Z 切片的二维半径邻域（默认 24 px），不是每 mm³ 细胞密度。融合表只是 `(mask pseudo-instance ID -> same mask-derived signal and spatial feature)` 的内连接；与外部作者蛋白 CSV 的数字 ID 完全独立。

## 四、待审核科学问题

1. Marker 原始数据是注册后 16 张 TIFF、一个多通道 3D 堆栈，还是某种作者汇总表达表？
2. 通道索引是否由作者文件元数据或正式 marker-to-channel 对照表确认？
3. 图像的 Z顺序是否与作者 mask 对应，不只是文件名看起来顺序一致？
4. 共同的 XY 像素坐标是否有实际几何配准证据，局部叠加是否存在明显系统性错位？
5. Instance mask 具体描绘核、细胞还是其他对象？如果是核，HER2/panCK/CD31/LYVE1 的细胞/膜表达怎样定量才有科学意义？

以上问题需要源数据验证，不可以从本模块生成的平均强度中反推证明。
