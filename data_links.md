# Image download & path mapping (Kaggle)

This repository does **not** include image files. It only distributes labels and structured tables (CSV/Parquet/Mat). Please download images from Kaggle.

## Kaggle datasets (images)
- dataset14 (dif-nih14, 14 classes): https://www.kaggle.com/datasets/yufaja/dif-nih14
- dataset15 (dif-orid5k-balanced, 8 classes): https://www.kaggle.com/datasets/yufaja/dif-orid5k-balanced
- dataset16 (dif-mritumor, 4 classes): https://www.kaggle.com/datasets/yufaja/dif-mritumor

## Extracted folder structure (no split)
All three Kaggle datasets do **not** provide train/val/test splits.

After extraction, the dataset folder (e.g., `dif-nih14/`) contains **all selected images directly under it**.

## How to use `image_path` in the tables
In each `dataset*/csv_data/pll_dataset.csv`:
- `image_path`: the image filename (UUID prefix removed; no subdirectory).

If you extracted the Kaggle images into `IMAGE_ROOT`, the local image path is:

`IMAGE_ROOT / image_path`

Example:
- If you have `.../dif-nih14/00000011_006.png`, then `IMAGE_ROOT=.../dif-nih14` and `image_path=00000011_006.png`.

## Optional: local trace path (maintainers only)
If you previously ran the upstream difficult-sample pipeline and kept the local image set (e.g., `~/ml4img/f4dficimg/...`), the trace fields in `pll_dataset.csv` include:
- `f4dficimg_path`: absolute path to the local difficult-image folder (for 100% traceability checks).

Note: `f4dficimg_path` is for internal auditing/traceability only; normal users can ignore it and use Kaggle + `image_path`.
