# Imprecise-Label Learning Datasets (SEU PLL Format)

本仓库用于将标注系统导出的结果（XLSX 快照）标准化导出为 **东南大学 PLL 数据集格式** 的表文件（CSV/Parquet）与 Matlab `.mat` 文件。

重要说明：
- **本仓库不包含任何图片文件**；图片来自我们在 `ml4img` 项目中筛选得到的困难样本子集，后续将托管到 Kaggle。
- **本仓库不提供 train/test 划分**：所有导出均保持原始完整数据形态，使用者可按自身实验设定自行划分。

## 数据来源与收集方法（困难样本筛选）
我们对原始公开数据集先训练 ResNet 并做 K 折交叉验证，再基于不确定性/分歧度聚合筛选困难样本，导出形成待标注图片集合（`f4dficimg`）。

筛选方法的实现与文档参考（上游仓库）：
- ML4md-img-cls（ml4img）：https://github.com/yufao/ML4md-img-cls
- 困难样本流程说明：`ml4img/doc/README_difficult_samples.md`
- 多标签困难样本聚合：`ml4img/src/scripts/aggregate_difficult_multilabel.py`
- 困难样本导出脚本：`ml4img/src/scripts/extract_difficult_images.py`

## 原图溯源与 UUID 前缀规则
标注系统导出的 `image_path` 形如：

`static/img/<uuid32>_<原文件名>`

溯源规则：
1. 去掉 `<uuid32>_` 前缀得到 `<原文件名>`；
2. 结合 `dataset_id ↔ f4dficimg 子目录` 的映射关系，定位到本地原图路径：

`~/ml4img/f4dficimg/<子目录>/<原文件名>`

映射关系见：`core_mappings/dataset_mapping.csv`。

补充说明（面向 Kaggle 使用者）：
- 对外分发/复现实验时，推荐直接从 Kaggle 下载图片；解压后的目录（如 `dif-nih14/`）下直接是全部图片文件（无任何划分）。
- 在导出表 `dataset*/csv_data/pll_dataset.csv` 中，`image_path` 已经是**去掉 UUID 前缀后的文件名**（不含子目录）。因此图片可按 `IMAGE_ROOT / image_path` 的方式定位。

## 目录结构
- `core_mappings/`
  - `dataset_mapping.csv`：dataset_id ↔ 子目录 ↔ 类别数
  - `image_id_mapping.csv`：image_id ↔ (uuid32, 原文件名, 溯源路径)
  - `label_dict.csv`：label_id ↔ label_name（按 dataset_id 拆分记录）
- `dataset14/`、`dataset15/`、`dataset16/`
  - `csv_data/`
    - `pll_dataset.csv`：主标注表（不含 train/test 字段），核心字段：
      - `image_id`：图片 ID（表内唯一）
      - `dataset_id`：数据集编号（14/15/16）
      - `image_path`：图片文件名（已去 UUID 前缀；用于 Kaggle/本地定位）
      - `partial_target`：偏标注候选集合（JSON list，元素为 `label_id`）
      - `target`：目标标签（单个 `label_id`）
      - 追溯字段（可用于回溯/调试）：`xlsx_image_path`、`uuid_prefix`、`f4dficimg_path`、`record_count`、`expert_ids_json`
    - `partial_target.parquet`：偏标注候选矩阵（snappy），形状为 $Q\times M$：
      - 行：`label_id`
      - 列：`image_id`
      - 值：0/1（是否属于候选集合）
    - `target.parquet`：目标标签向量（snappy），形状为 $1\times M$：
      - 列：`image_id`
      - 值：`label_id`
  - `mat_data/pll_dataset.mat`：Matlab 可直接 `load()`（不含 trainIndex/testIndex），包含 `data/partial_target/target/image_ids`
  - `id_mapping.csv`：该数据集内的溯源映射表（含 UUID 前缀、原文件名、本地溯源路径等）
  - `data_stats.md`：自动统计报告（样本数、标签分布、偏标注歧义度等）
  - `code/`：工具脚本（无划分逻辑）
    - `data_loader.py`：加载 `csv_data/` 与 `mat_data/`（SQLite 为可选补充）
    - `stats_visualization.py`：简单可视化（例如标签分布）
- `batch_processor.py`：一键导出与一致性校验（导出/只校验两种模式）
- `audit_selfcheck.py`：自查清单脚本，生成 `audit_report.md` 与 `audit_logs/` 日志
- `audit_logs/`：导出与自查过程日志（便于复现与审计）

## 数据获取（本地路径 / Kaggle 入口）
- 本仓库不包含图片，也不要求使用者具备 `ml4img` 本地图片目录。
- 图片下载入口与映射说明见：`data_links.md`：
  - dataset14（dif-nih14）：https://www.kaggle.com/datasets/yufaja/dif-nih14
  - dataset15（dif-orid5k-balanced）：https://www.kaggle.com/datasets/yufaja/dif-orid5k-balanced
  - dataset16（dif-mritumor）：https://www.kaggle.com/datasets/yufaja/dif-mritumor

最常见用法（Kaggle 下载后直接用）：
1. 下载并解压（例如得到 `.../dif-nih14/`，里面直接是全部图片文件，无划分）；
2. 使用 `dataset14/csv_data/pll_dataset.csv` 的 `image_path` 拼接出本地图片路径：
   - `full_path = IMAGE_ROOT / image_path`

## 一键导出与校验
在仓库根目录执行：

```bash
python batch_processor.py --export
python batch_processor.py --validate-only
```

导出过程将强制校验：`image_id → f4dficimg` 溯源 **100% 命中**，若有任何图片缺失会直接报错停止。

## 许可说明（代码 vs 数据）
- 代码与文档：见 `LICENSE`（当前为 Apache-2.0）。
- 数据集图片与原始数据集的许可：请以原始公开数据集/平台协议为准；本仓库仅分发标签与结构化表。
- 分数据集许可对齐约定见：`LICENSE_DATASETS.md`（用于后续 Kaggle 页面与论文/仓库说明的口径）。

（dataset14/15/16 的上游链接与许可证原文链接将在后续补充到 `data_links.md`。）
