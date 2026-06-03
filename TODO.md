# TODO: 真实数据接入与合成流程替换

本文档用于梳理当前 UnfavorSeg 流程中由 dense label 合成/模拟的数据，并按优先级列出需要补充的真实数据格式、代码改造和验证任务。

## P0 - 必须优先完成

### 1. 明确并接入真实三维体数据

目标：建立真实的 nnU-Net 数据集输入，替代仅用于 smoke/demo 的数据假设。

所需数据：

- `vp` 三维体数据：`imagesTr/<case_id>_0000.nii.gz`
- `vs` 三维体数据：`imagesTr/<case_id>_0001.nii.gz`
- `burial depth` 三维体数据：`imagesTr/<case_id>_0002.nii.gz`
- 可选但强烈建议：真实 dense label：`labelsTr/<case_id>.nii.gz`
- 每个体数据必须保留一致的 `spacing`、`origin`、`direction/affine`

类别编码建议：

```text
0 = background
1 = fracture_zone
2 = soft_rock
3 = water_rich_zone
```

待办：

- 确认三类地质标签是互斥单标签，还是允许重叠的多标签。
- 确认所有 case 的 `vp/vs/depth/label` shape 和空间几何信息完全一致。
- 生成或校验 `dataset.json`。

相关代码：

- `segment/data/make_dataset.py`
- `segment/io.py`

### 2. 接入真实 TFR / 掌子面记录表

当前问题：`segment/experiments/records.py` 会从 dense label 中分层抽样 voxel 来模拟粗分类器训练记录。这不能替代真实 TFR 表。

推荐文件：

```text
tfr_records.csv
```

最低字段：

```text
record_id
project_id
tunnel_id
case_id
chainage_m
class_label
split
is_used_for_training
```

推荐补充字段：

```text
section_id
x
y
z
chainage_start_m
chainage_end_m
local_y_min
local_y_max
local_z_min
local_z_max
confidence
source_type
observer
record_date
```

待办：

- 新增真实 TFR CSV 读取器。
- 将 TFR 里程/坐标映射到 voxel grid。
- 替换 `records.py` 中的 dense label 抽样逻辑。
- 保证 `split=test/val` 的记录不参与 RF 训练和伪标签生成。

相关代码：

- `segment/experiments/records.py`
- `segment/data/alignment.py`
- `segment/cli.py`

### 3. 接入真实钻孔 / 探孔日志

当前问题：`e3_borehole.py` 使用 `boreholes_from_label()` 从 dense label 随机抽取 1D 轨迹，不是真实钻孔数据。

推荐文件：

```text
borehole_intervals.csv
```

最低字段：

```text
borehole_id
project_id
tunnel_id
case_id
x0
y0
z0
x1
y1
z1
interval_start_m
interval_end_m
class_label
```

更推荐字段：

```text
borehole_id
project_id
tunnel_id
case_id
collar_x
collar_y
collar_z
azimuth_deg
dip_deg
depth_start_m
depth_end_m
interval_start_m
interval_end_m
class_label
lithology_text
confidence
source
record_date
```

待办：

- 新增钻孔/探孔 CSV 读取器。
- 根据孔口坐标、方位角、倾角和区间深度生成真实 3D 轨迹。
- 将轨迹采样到 voxel grid。
- 替换 `validation_records.py` 中的 `boreholes_from_label()` 合成逻辑。
- 修改 `e3_borehole.py`，从真实 interval 表计算 F1、hit rate 和 boundary error。

相关代码：

- `segment/experiments/e3_borehole.py`
- `segment/experiments/validation_records.py`
- `segment/experiments/eval_metrics.py`
- `segment/data/alignment.py`

## P1 - 重要增强

### 4. 接入真实 held-out TFR 验证记录

当前问题：`e2_tfr_finegrained.py` 使用 `faces_from_label()` 从 reference label 里抽取轴向切片，模拟 held-out TFR。

待办：

- 使用真实 `tfr_records.csv` 中 `split=val/test` 的记录。
- 按 `chainage_m` 或真实坐标从预测体中采样。
- 保留 `tolerance_sections`，但应基于真实里程误差或 voxel spacing 换算。
- 替换 `faces_from_label()` 合成流程。

相关代码：

- `segment/experiments/e2_tfr_finegrained.py`
- `segment/experiments/validation_records.py`

### 5. 明确空间配准输入格式

推荐文件：

```text
centerline.csv
dem.tif / dem.csv / dem.npy
```

`centerline.csv` 字段：

```text
project_id
tunnel_id
chainage_m
x
y
z
```

DEM 数据至少提供：

```text
x
y
surface_z
coordinate_system
```

待办：

- 明确项目使用的坐标系。
- 建立 `chainage_m -> world xyz -> voxel index` 的标准转换。
- 用真实 DEM 生成 `burial depth` 通道。
- 记录每个 case 的 `Mg` 或 affine 信息，保证 TFR/钻孔/体数据可互相映射。

相关代码：

- `segment/data/alignment.py`

### 6. 区分真实标签、伪标签和模型预测

当前容易混淆的内容：

- `labelsTr/<case>.nii.gz`：真实 dense label，若存在
- `_pseudolabels_<dataset>/<case>.nii.gz`：RF 生成的 hard pseudo-label
- `_pseudolabels_<dataset>/prob_<case>.nii.gz`：RF 生成的 confidence
- `results/pred_*`：fine model 预测结果

待办：

- 在 README 和输出目录命名中明确区分这些来源。
- 所有实验表格增加 `data_source` 或备注，标明是真实记录还是 synthetic fallback。
- 禁止在论文结果中把 synthetic fallback 说成独立现场验证。

相关代码：

- `segment/coarse/pseudolabel.py`
- `segment/fine/dataset.py`
- `segment/experiments/uncertainty.py`
- `README.md`

## P2 - 后续完善

### 7. 保留 synthetic fallback，但只能用于 demo/smoke

当前 synthetic 来源：

- `faces_from_label()`：从 dense label 合成 TFR face
- `boreholes_from_label()`：从 dense label 合成钻孔轨迹
- `records.py`：从 dense label 抽样模拟粗分类器训练记录

待办：

- 增加显式参数，例如 `--synthetic-records`。
- 默认优先使用真实 CSV。
- 如果没有真实 CSV，CLI 应打印警告：当前结果仅用于 demo/smoke，不可作为独立现场验证。

相关代码：

- `segment/experiments/validation_records.py`
- `segment/experiments/records.py`
- `segment/cli.py`

### 8. 完善鲁棒性与扰动实验说明

当前 `ab_tsp_perturbation.py` 中的 `Vp/Vs +/-3/5/10%` 是人为合成扰动，不是新采集数据。

待办：

- 在实验表格和 README 中明确标注为 synthetic perturbation。
- 若有真实重复采集或反演不确定性分布，应替换固定比例扰动。
- 输出扰动水平、扰动来源和解释说明。

相关代码：

- `segment/experiments/ab_tsp_perturbation.py`
- `configs/geology.yaml`

### 9. 增加数据质量检查脚本

待办：

- 检查 volume shape/spacing/origin/direction 是否一致。
- 检查 TFR 记录是否能映射到有效 voxel。
- 检查钻孔轨迹是否落在 volume 范围内。
- 检查类别编码是否在允许范围内。
- 检查 train/val/test 是否泄漏。

建议新增：

```text
segment data-check --dataset <dataset> --tfr tfr_records.csv --boreholes borehole_intervals.csv
```

## 当前合成内容清单

以下内容当前不是来自真实现场记录：

| 内容 | 当前来源 | 代码位置 | 是否可用于正式独立验证 |
|---|---|---|---|
| 粗分类器训练 records | dense label 分层 voxel 抽样 | `segment/experiments/records.py` | 否 |
| held-out TFR face | reference label 轴向切片 | `segment/experiments/validation_records.py` | 否 |
| 钻孔/探孔轨迹 | reference label 随机 1D 轨迹 | `segment/experiments/validation_records.py` | 否 |
| hard pseudo-label | RF 全体 voxel 滑窗预测 | `segment/coarse/pseudolabel.py` | 可作为弱监督，不是真实标签 |
| confidence 通道 | RF 最大类别概率 | `segment/coarse/pseudolabel.py` | 可作为弱监督，不是真实标签 |
| TSP 扰动实验 | 人为 `Vp/Vs +/-3/5/10%` | `segment/experiments/ab_tsp_perturbation.py` | 仅为敏感性分析 |

## 验收标准

- 真实 TFR CSV 和钻孔 CSV 能被代码读取并映射到 voxel grid。
- `e2_tfr_finegrained.py` 不再默认从 dense label 合成 TFR 验证记录。
- `e3_borehole.py` 不再默认从 dense label 合成钻孔轨迹。
- 粗阶段 RF 训练可以选择真实 TFR records，而不是只能 dense label 抽样。
- 所有报告中明确标注数据来源：real / pseudo-label / synthetic fallback。
