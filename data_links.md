# 数据下载入口（Kaggle）

本仓库不包含任何图片文件；仅分发标签与结构化表（CSV/Parquet/Mat）。图片请从 Kaggle 获取。

## Kaggle 数据集链接
- dataset14（dif-nih14，14 类）：https://www.kaggle.com/datasets/yufaja/dif-nih14
- dataset15（dif-orid5k-balanced，8 类）：https://www.kaggle.com/datasets/yufaja/dif-orid5k-balanced
- dataset16（dif-mritumor，4 类）：https://www.kaggle.com/datasets/yufaja/dif-mritumor

## Kaggle 解压后的目录结构（无划分）
三套 Kaggle 数据集均不包含任何 train/test 划分。

使用者从 Kaggle 下载并解压后，目录（例如 `dif-nih14/`）下直接是本次筛选出来的全部图片文件。

## 表文件中的路径字段如何使用
在每个 `dataset*/csv_data/pll_dataset.csv` 中：
- `image_path`：图片文件名（不含 UUID 前缀，不含子目录）。

因此，当你把 Kaggle 数据集解压到某个目录 `IMAGE_ROOT` 后，图片的本地路径为：

`IMAGE_ROOT / image_path`

示例：
- 若解压后得到 `.../dif-nih14/00000011_006.png`，则 `IMAGE_ROOT=.../dif-nih14`，`image_path=00000011_006.png`。

## 可选：本地溯源路径（仅对内部复现/回溯有用）
若你本地曾运行过上游困难样本筛选并保留了图片集合（例如 `~/ml4img/f4dficimg/...`），则 `pll_dataset.csv` 的追溯字段里会包含：
- `f4dficimg_path`：该图片在本地困难样本目录中的绝对路径（用于 100% 溯源校验）。

> 注意：`f4dficimg_path` 是追溯信息，不要求使用者具备该目录；对外分发与复现实验推荐按 Kaggle 下载路径使用 `image_path`。
