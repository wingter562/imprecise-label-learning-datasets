from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _dataset_root() -> Path:
    return Path(__file__).resolve().parents[1]


def plot_label_distribution(
    dataset_root: Optional[Path] = None,
    which: str = "target",
    out_png: Optional[Path] = None,
) -> None:
    """生成标签分布柱状图（不包含 train/test 划分相关统计）。"""
    root = dataset_root or _dataset_root()
    if which == "target":
        # target.parquet: 1×M，index=target_label_id，columns=image_id
        df = pd.read_parquet(root / "csv_data" / "target.parquet")
        row = df.iloc[0]
        counts = row.value_counts().sort_index()
        labels = [str(int(k)) for k in counts.index.tolist()]
        values = counts.values
    else:
        # partial_target.parquet: Q×M，index=label_id，columns=image_id
        df = pd.read_parquet(root / "csv_data" / "partial_target.parquet")
        labels = [str(x) for x in df.index.tolist()]
        values = df.sum(axis=1).astype(int).values

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 4))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("count")
    ax.set_title(f"{root.name}: {which} label distribution")
    fig.tight_layout()

    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["target", "partial"], default="target")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    which = "target" if args.which == "target" else "partial"
    out = Path(args.out) if args.out else None
    plot_label_distribution(which=which, out_png=out)


if __name__ == "__main__":
    main()
