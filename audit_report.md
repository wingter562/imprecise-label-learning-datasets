# 标准化导出自查报告

- 时间: 2026-03-23T22:04:37
- 通过项数: 13/13

## 检查结果
- 核心映射表完整性: 通过 — core_mappings 三表存在、可读、非空、无 NaN，且 dataset_id 映射符合要求
- 目录结构与文档: 通过 — 根目录与 dataset14/15/16 结构齐全，核心文件均非空
- dataset14 CSV/Parquet 内容: 通过 — pll_dataset.csv 字段齐全，Parquet 维度/取值/label_id 范围均正确
- dataset14 溯源映射: 通过 — id_mapping 字段齐全，且 f4dficimg_path 100% 存在
- dataset14 Mat 兼容性: 通过 — Mat 字段齐全、维度合理，且不含 train/test 划分字段
- dataset15 CSV/Parquet 内容: 通过 — pll_dataset.csv 字段齐全，Parquet 维度/取值/label_id 范围均正确
- dataset15 溯源映射: 通过 — id_mapping 字段齐全，且 f4dficimg_path 100% 存在
- dataset15 Mat 兼容性: 通过 — Mat 字段齐全、维度合理，且不含 train/test 划分字段
- dataset16 CSV/Parquet 内容: 通过 — pll_dataset.csv 字段齐全，Parquet 维度/取值/label_id 范围均正确
- dataset16 溯源映射: 通过 — id_mapping 字段齐全，且 f4dficimg_path 100% 存在
- dataset16 Mat 兼容性: 通过 — Mat 字段齐全、维度合理，且不含 train/test 划分字段
- CSV 编码/格式: 通过 — /home/droot/pll_dts/imprecise-label-learning-datasets/dataset14/csv_data/pll_dataset.csv: text/csv; charset=us-ascii（us-ascii 为 UTF-8 子集，UTF-8 解码通过）
- 可追溯性文档: 通过 — README 与 data_stats.md 包含核心来源/方法/不划分说明

## 核心统计摘要
- dataset14: M=4683 Q=14
- dataset15: M=2765 Q=8
- dataset16: M=1431 Q=4
