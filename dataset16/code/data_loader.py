from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from scipy.io import loadmat


def _dataset_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_csv_parquet(dataset_root: Optional[Path] = None) -> Dict[str, Any]:
    """加载本数据集的 CSV/Parquet 产物（不包含任何 train/test 划分逻辑）。"""
    root = dataset_root or _dataset_root()
    pll = pd.read_csv(root / "csv_data" / "pll_dataset.csv")
    partial_target = pd.read_parquet(root / "csv_data" / "partial_target.parquet")
    target = pd.read_parquet(root / "csv_data" / "target.parquet")

    if "partial_target" in pll.columns:
        pll["partial_target_list"] = pll["partial_target"].apply(lambda s: json.loads(s) if isinstance(s, str) and s else [])
    if "expert_ids_json" in pll.columns:
        pll["expert_ids"] = pll["expert_ids_json"].apply(lambda s: json.loads(s) if isinstance(s, str) and s else [])

    return {
        "pll": pll,
        "partial_target": partial_target,
        "target": target,
    }


def load_mat(dataset_root: Optional[Path] = None) -> Dict[str, Any]:
    """加载 Matlab .mat（Matlab R2020b+ 可直接 load）。"""
    root = dataset_root or _dataset_root()
    mat = loadmat(root / "mat_data" / "pll_dataset.mat")

    def _squeeze(x):
        try:
            return x.squeeze()
        except Exception:
            return x

    return {
        "data": _squeeze(mat.get("data")),
        "partial_target": mat.get("partial_target"),
        "target": mat.get("target"),
        "image_ids": _squeeze(mat.get("image_ids")),
    }


def load_sqlite(dataset_root: Optional[Path] = None) -> Tuple[sqlite3.Connection, Dict[str, pd.DataFrame]]:
    """SQLite 为可选补充产物；当前若不存在会抛错。"""
    root = dataset_root or _dataset_root()
    db_path = root / "sqlite" / "pll_dataset.db"
    conn = sqlite3.connect(str(db_path))
    tables = {}
    for name in ["pll_dataset", "partial_target", "target"]:
        try:
            tables[name] = pd.read_sql_query(f"SELECT * FROM {name}", conn)
        except Exception:
            pass
    return conn, tables
