#!/usr/bin/env python3
"""Batch export for PLL datasets (dataset14/15/16).

核心目标：
- 从 medc-img-annotation-app 导出的 XLSX（images/annotations/labels/datasets）中，
  生成：core_mappings + dataset14/15/16 的 CSV/Parquet + Mat（不包含 train/test 划分字段）。
- 不拷贝/不修改任何原图，仅生成可溯源的路径引用（~/ml4img/f4dficimg/...）。

使用：
  python batch_processor.py --export
  python batch_processor.py --validate-only

仅更新指定数据集（不会触碰其它数据集产物；会对 core_mappings 做同 dataset_id 的局部更新）：
    python batch_processor.py --export --only-datasets 14 --xlsx14 /path/to/ds14.xlsx

默认读取 res_src 中 20260319 的 3 份 XLSX；如需替换可通过参数覆盖。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.io import savemat


RE_UUID_PREFIX = re.compile(r"^static/img/(?P<uuid>[0-9a-f]{32})_(?P<fname>.+)$")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: int
    name: str
    f4dficimg_subdir: str
    expected_num_classes: int
    xlsx_path: Path
    out_dir: Path


def truthy(v: object) -> bool:
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}


def parse_label_ids(raw: object) -> List[int]:
    """解析 XLSX 的 label_ids 字段。

已观测到格式：
- 字符串："[88, 87, 81]" 或 "[93]"
- 缺失：NaN
"""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        out: List[int] = []
        for x in raw:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out

    s = str(raw).strip()
    if not s:
        return []

    # JSON/py-list 风格
    if s.startswith("[") and s.endswith("]"):
        # 兼容空格
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                continue
        return out

    # 兜底：按非数字分隔提取
    nums = re.findall(r"\d+", s)
    return [int(x) for x in nums]


def split_uuid_and_filename(image_path: str) -> Tuple[str, str]:
    """从 static/img/<uuid32>_<原文件名> 中提取 uuid 与原文件名。

返回 (uuid32, original_filename)。若不匹配则 uuid32 为空、original_filename 为 basename。
"""
    m = RE_UUID_PREFIX.match(str(image_path).strip())
    if not m:
        p = Path(str(image_path))
        return "", p.name
    return m.group("uuid"), m.group("fname")


def expand_f4dficimg_path(subdir: str, original_filename: str) -> Tuple[str, Path]:
    """生成需要写入表文件的路径字符串（带 ~），以及用于校验的实际 Path。"""
    logical = f"~/ml4img/f4dficimg/{subdir}/{original_filename}"
    physical = Path(logical).expanduser()
    return logical, physical


def read_xlsx(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取单个 dataset 的 XLSX 并返回 (ann, images, labels, ds_info)。"""
    xf = pd.ExcelFile(spec.xlsx_path)

    # 通过包含关键字识别 sheet，避免未来改名
    def pick_sheet(keys: Sequence[str]) -> str:
        for s in xf.sheet_names:
            if all(k in s for k in keys):
                return s
        raise KeyError(f"cannot find sheet by keys={keys} in {xf.sheet_names}")

    ann_sheet = pick_sheet(["标注"])
    img_sheet = pick_sheet(["图片"])
    label_sheet = pick_sheet(["标签"])
    info_sheet = pick_sheet(["信息"])

    ann = pd.read_excel(xf, sheet_name=ann_sheet)
    images = pd.read_excel(xf, sheet_name=img_sheet)
    labels = pd.read_excel(xf, sheet_name=label_sheet)
    info = pd.read_excel(xf, sheet_name=info_sheet)

    # 统一类型
    ann["dataset_id"] = ann["dataset_id"].astype(int)
    ann["image_id"] = ann["image_id"].astype(int)
    ann["label_id"] = ann["label_id"].astype(int)
    images["image_id"] = images["image_id"].astype(int)
    labels["label_id"] = labels["label_id"].astype(int)

    return ann, images, labels, info


def build_core_mappings(specs: Sequence[DatasetSpec]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成并返回三张 core_mappings DataFrame。"""
    dataset_rows = []
    image_rows = []
    label_rows = []

    for spec in specs:
        ann, images, labels, _info = read_xlsx(spec)

        dataset_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "f4dficimg_subdir": spec.f4dficimg_subdir,
                "num_classes": int(spec.expected_num_classes),
                "f4dficimg_dir": f"~/ml4img/f4dficimg/{spec.f4dficimg_subdir}",
            }
        )

        # image_id_mapping：以 images 表为基准（全量图片），补全 uuid/orig/f4dficimg 溯源路径
        for _, r in images.iterrows():
            image_id = int(r["image_id"])
            xlsx_image_path = str(r["image_path"])
            uuid_prefix, orig_fname = split_uuid_and_filename(xlsx_image_path)
            logical, physical = expand_f4dficimg_path(spec.f4dficimg_subdir, orig_fname)
            if not physical.exists():
                raise FileNotFoundError(
                    f"溯源失败：dataset_id={spec.dataset_id} image_id={image_id} 原文件名={orig_fname} 期望路径={physical}"
                )
            image_rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "image_id": image_id,
                    "xlsx_image_path": xlsx_image_path,
                    "uuid_prefix": uuid_prefix,
                    "original_filename": orig_fname,
                    "f4dficimg_path": logical,
                }
            )

        # label_dict：来自 labels 表
        for _, r in labels.iterrows():
            label_rows.append(
                {
                    "dataset_id": spec.dataset_id,
                    "label_id": int(r["label_id"]),
                    "label_name": str(r.get("label_name", "")),
                    "category": str(r.get("category", "")),
                    "description": "",
                }
            )

    dataset_mapping = pd.DataFrame(dataset_rows).sort_values(["dataset_id"]).reset_index(drop=True)
    image_id_mapping = pd.DataFrame(image_rows).sort_values(["dataset_id", "image_id"]).reset_index(drop=True)
    label_dict = pd.DataFrame(label_rows).sort_values(["dataset_id", "label_id"]).reset_index(drop=True)

    # 完整性：禁止缺失
    for name, df in [("dataset_mapping", dataset_mapping), ("image_id_mapping", image_id_mapping), ("label_dict", label_dict)]:
        if df.isna().any().any():
            bad = df.isna().sum().to_dict()
            raise ValueError(f"{name} 存在缺失值: {bad}")

    return dataset_mapping, image_id_mapping, label_dict


def make_label_order(labels_df: pd.DataFrame) -> List[int]:
    ids = sorted({int(x) for x in labels_df["label_id"].tolist()})
    return ids


def parse_only_datasets(raw: str) -> Optional[Set[int]]:
    s = (raw or "").strip()
    if not s:
        return None
    out: Set[int] = set()
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out.add(int(p))
    return out or None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_core_mappings_for_spec(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """仅基于单个 spec 的 XLSX 生成其对应的三张映射表片段。

    注意：此处会做本地 f4dficimg_path 存在性校验（维护者/溯源用途）。
    """
    ann, images, labels, _info = read_xlsx(spec)

    dataset_mapping = pd.DataFrame(
        [
            {
                "dataset_id": spec.dataset_id,
                "f4dficimg_subdir": spec.f4dficimg_subdir,
                "num_classes": int(spec.expected_num_classes),
                "f4dficimg_dir": f"~/ml4img/f4dficimg/{spec.f4dficimg_subdir}",
            }
        ]
    )

    # image_id_mapping：以 images 表为基准（全量图片），补全 uuid/orig/f4dficimg 溯源路径
    images = images.copy()
    images["image_id"] = images["image_id"].astype(int)
    images["image_path"] = images["image_path"].astype(str)

    img_rows: List[dict] = []
    for _, r in images.iterrows():
        image_id = int(r["image_id"])
        xlsx_image_path = str(r["image_path"])
        uuid_prefix, orig_fname = split_uuid_and_filename(xlsx_image_path)
        logical, physical = expand_f4dficimg_path(spec.f4dficimg_subdir, orig_fname)
        if not physical.exists():
            raise FileNotFoundError(
                f"溯源失败：dataset_id={spec.dataset_id} image_id={image_id} 原文件名={orig_fname} 期望路径={physical}"
            )
        img_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "image_id": image_id,
                "xlsx_image_path": xlsx_image_path,
                "uuid_prefix": uuid_prefix,
                "original_filename": orig_fname,
                "f4dficimg_path": logical,
            }
        )
    image_id_mapping = pd.DataFrame(img_rows).sort_values(["dataset_id", "image_id"]).reset_index(drop=True)

    # label_dict：来自 labels 表
    label_rows: List[dict] = []
    for _, r in labels.iterrows():
        label_rows.append(
            {
                "dataset_id": spec.dataset_id,
                "label_id": int(r["label_id"]),
                "label_name": str(r.get("label_name", "")),
                "category": str(r.get("category", "")),
                "description": "",
            }
        )
    label_dict = pd.DataFrame(label_rows).sort_values(["dataset_id", "label_id"]).reset_index(drop=True)

    # 完整性：禁止缺失
    for name, df in [
        ("dataset_mapping", dataset_mapping),
        ("image_id_mapping", image_id_mapping),
        ("label_dict", label_dict),
    ]:
        if df.isna().any().any():
            bad = df.isna().sum().to_dict()
            raise ValueError(f"{name} 存在缺失值: {bad}")

    # dataset_id 一致性检查：labels 的类别数要匹配预期
    q = len(make_label_order(labels))
    if q != spec.expected_num_classes:
        raise ValueError(f"dataset_id={spec.dataset_id} 类别数不匹配：labels表={q} 期望={spec.expected_num_classes}")

    return dataset_mapping, image_id_mapping, label_dict


def update_core_mappings_partial(repo_root: Path, specs_to_update: Sequence[DatasetSpec], all_specs: Sequence[DatasetSpec]) -> None:
    """只更新 core_mappings 中指定 dataset_id 的行。

    - 若 core_mappings 不存在：退化为全量重建（all_specs）
    - 若存在：对每个 spec 读取其 XLSX，生成片段并替换对应 dataset_id
    """
    core = repo_root / "core_mappings"
    _ensure_dir(core)

    dm_path = core / "dataset_mapping.csv"
    im_path = core / "image_id_mapping.csv"
    ld_path = core / "label_dict.csv"

    if not (dm_path.exists() and im_path.exists() and ld_path.exists()):
        dataset_mapping, image_id_mapping, label_dict = build_core_mappings(list(all_specs))
        dm_path.write_text(dataset_mapping.to_csv(index=False), encoding="utf-8")
        im_path.write_text(image_id_mapping.to_csv(index=False), encoding="utf-8")
        ld_path.write_text(label_dict.to_csv(index=False), encoding="utf-8")
        return

    # load existing
    dataset_mapping = pd.read_csv(dm_path, keep_default_na=False)
    image_id_mapping = pd.read_csv(im_path, keep_default_na=False)
    label_dict = pd.read_csv(ld_path, keep_default_na=False)

    for spec in specs_to_update:
        dm_new, im_new, ld_new = build_core_mappings_for_spec(spec)
        dataset_mapping = pd.concat([dataset_mapping[dataset_mapping["dataset_id"] != spec.dataset_id], dm_new], ignore_index=True)
        image_id_mapping = pd.concat([image_id_mapping[image_id_mapping["dataset_id"] != spec.dataset_id], im_new], ignore_index=True)
        label_dict = pd.concat([label_dict[label_dict["dataset_id"] != spec.dataset_id], ld_new], ignore_index=True)

    dataset_mapping = dataset_mapping.sort_values(["dataset_id"]).reset_index(drop=True)
    image_id_mapping = image_id_mapping.sort_values(["dataset_id", "image_id"]).reset_index(drop=True)
    label_dict = label_dict.sort_values(["dataset_id", "label_id"]).reset_index(drop=True)

    dm_path.write_text(dataset_mapping.to_csv(index=False), encoding="utf-8")
    im_path.write_text(image_id_mapping.to_csv(index=False), encoding="utf-8")
    ld_path.write_text(label_dict.to_csv(index=False), encoding="utf-8")


def export_one_dataset(spec: DatasetSpec) -> None:
    ann, images, labels, _info = read_xlsx(spec)

    # 只导出本数据集的标注（以 ann 为权威）
    ann = ann[ann["dataset_id"] == spec.dataset_id].copy()
    if ann.empty:
        raise ValueError(f"XLSX 中 dataset_id={spec.dataset_id} 的标注为空: {spec.xlsx_path}")

    # images join
    images = images.copy()
    images["image_path"] = images["image_path"].astype(str)

    # label order
    label_order = make_label_order(labels)
    q = len(label_order)
    if q != spec.expected_num_classes:
        raise ValueError(
            f"dataset_id={spec.dataset_id} 类别数不匹配：labels表={q} 期望={spec.expected_num_classes}"
        )
    label_to_row = {lid: i for i, lid in enumerate(label_order)}

    # 以标注出现的 image_id 为样本集（避免把未标注图片混入）
    image_ids = sorted({int(x) for x in ann["image_id"].tolist()})

    # 校验：每个 image_id 都能在 images 表中找到路径
    img_map = dict(zip(images["image_id"].astype(int).tolist(), images["image_path"].astype(str).tolist()))
    missing_in_images = [iid for iid in image_ids if iid not in img_map]
    if missing_in_images:
        raise KeyError(f"dataset_id={spec.dataset_id} 标注中有 image_id 不在图片表中: {missing_in_images[:10]}...")

    m = len(image_ids)
    partial_target = np.zeros((q, m), dtype=np.int8)
    target = np.zeros((q, m), dtype=np.int8)

    # 统计：多次标注/一致性
    per_image_records: Dict[int, List[dict]] = {iid: [] for iid in image_ids}

    for _, r in ann.iterrows():
        iid = int(r["image_id"])
        rec = {
            "record_id": int(r.get("record_id")) if "record_id" in r and not pd.isna(r.get("record_id")) else None,
            "expert_id": str(r.get("expert_id", "")),
            "label_id": int(r["label_id"]),
            "label_ids": parse_label_ids(r.get("label_ids")),
        }
        per_image_records[iid].append(rec)

    # 多次标注一致性指标（基于 per_image_records 的真实记录）
    multi_images = [iid for iid, recs in per_image_records.items() if len(recs) > 1]
    multi_image_rate = float(len(multi_images) / max(1, len(image_ids)))
    label_agree_flags: List[bool] = []
    partial_jaccards: List[float] = []
    for iid in multi_images:
        recs = per_image_records[iid]
        lbls = [int(x["label_id"]) for x in recs]
        label_agree_flags.append(len(set(lbls)) == 1)

        sets: List[Set[int]] = []
        for rec in recs:
            lids = rec.get("label_ids") or []
            if not lids:
                lids = [int(rec["label_id"])]
            sets.append(set(int(x) for x in lids))

        # 平均两两 Jaccard
        if len(sets) >= 2:
            pair_vals = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    a, b = sets[i], sets[j]
                    inter = len(a & b)
                    union = len(a | b)
                    pair_vals.append(float(inter / union) if union else 1.0)
            if pair_vals:
                partial_jaccards.append(float(np.mean(pair_vals)))

    label_agree_rate = float(np.mean(label_agree_flags)) if label_agree_flags else None
    partial_jaccard_mean = float(np.mean(partial_jaccards)) if partial_jaccards else None

    rows_for_table = []
    idmap_rows = []
    target_label_vec: List[int] = []

    for col, iid in enumerate(image_ids):
        xlsx_image_path = img_map[iid]
        uuid_prefix, orig_fname = split_uuid_and_filename(xlsx_image_path)
        logical_src, physical_src = expand_f4dficimg_path(spec.f4dficimg_subdir, orig_fname)
        if not physical_src.exists():
            raise FileNotFoundError(f"溯源失败：dataset_id={spec.dataset_id} image_id={iid} 路径={physical_src}")

        recs = per_image_records.get(iid, [])
        if not recs:
            raise ValueError(f"dataset_id={spec.dataset_id} image_id={iid} 没有任何标注记录")

        # 规则：target=label_id（单标签真值）；partial=label_ids（候选集合）
        # 若存在多条记录：target 取 label_id 的多数票（平票则取最后一条）；
        # 若 label_ids 为空：退化为 [label_id]
        vote = {}
        for rec in recs:
            lid = int(rec["label_id"])
            vote[lid] = vote.get(lid, 0) + 1
        max_cnt = max(vote.values())
        top = [lid for lid, c in vote.items() if c == max_cnt]
        target_label_id = int(recs[-1]["label_id"]) if len(top) > 1 else int(top[0])
        partial_set: Set[int] = set()
        for rec in recs:
            lids = rec.get("label_ids") or []
            if not lids:
                lids = [int(rec["label_id"])]
            partial_set.update(int(x) for x in lids)

        if target_label_id not in partial_set:
            partial_set.add(target_label_id)

        # 填矩阵
        if target_label_id not in label_to_row:
            raise KeyError(f"dataset_id={spec.dataset_id} image_id={iid} target_label_id={target_label_id} 不在 label_dict")
        target[label_to_row[target_label_id], col] = 1

        for lid in sorted(partial_set):
            if lid not in label_to_row:
                raise KeyError(f"dataset_id={spec.dataset_id} image_id={iid} partial label_id={lid} 不在 label_dict")
            partial_target[label_to_row[lid], col] = 1

        target_label_vec.append(int(target_label_id))

        # 对齐自查清单：
        # - image_path：去 UUID 后的原文件名（不含 static/img 前缀）
        # - partial_target：偏标注候选集合（JSON list 字符串）
        # - target：真实标签（label_id）
        rows_for_table.append(
            {
                "dataset_id": spec.dataset_id,
                "image_id": iid,
                "image_path": orig_fname,
                "partial_target": json.dumps(sorted(partial_set), ensure_ascii=False),
                "target": int(target_label_id),
                # 附加信息（非必需字段，但用于追溯/复核）
                "xlsx_image_path": xlsx_image_path,
                "uuid_prefix": uuid_prefix,
                "f4dficimg_path": logical_src,
                "record_count": len(recs),
                "expert_ids_json": json.dumps(
                    sorted({r.get("expert_id", "") for r in recs if str(r.get("expert_id", "")).strip()}),
                    ensure_ascii=False,
                ),
            }
        )

        idmap_rows.append(
            {
                "image_id": iid,
                "uuid_prefix": uuid_prefix,
                "original_filename": orig_fname,
                "f4dficimg_path": logical_src,
                "xlsx_image_path": xlsx_image_path,
            }
        )

    # 写 id_mapping.csv
    id_mapping_df = pd.DataFrame(idmap_rows)
    id_mapping_df.to_csv(spec.out_dir / "id_mapping.csv", index=False, encoding="utf-8")

    # 写 pll_dataset.csv
    pll_df = pd.DataFrame(rows_for_table).sort_values("image_id").reset_index(drop=True)
    pll_df.to_csv(spec.out_dir / "csv_data" / "pll_dataset.csv", index=False, encoding="utf-8")

    # 写 parquet：对齐自查清单
    # - partial_target.parquet：Q×M 矩阵（0/1）
    # - target.parquet：1×M label_id 向量
    pt_df = pd.DataFrame(partial_target.astype(np.int8), index=[int(x) for x in label_order], columns=[int(x) for x in image_ids])
    pt_df.index.name = "label_id"
    pt_df.to_parquet(spec.out_dir / "csv_data" / "partial_target.parquet", index=True, engine="pyarrow", compression="snappy")

    tgt_row = pd.DataFrame([target_label_vec], index=["target_label_id"], columns=[int(x) for x in image_ids])
    tgt_row.to_parquet(spec.out_dir / "csv_data" / "target.parquet", index=True, engine="pyarrow", compression="snappy")

    # 写 Mat（兼容 Matlab R2020b+）
    mat_payload = {
        "data": np.array(image_ids, dtype=np.int64).reshape(-1, 1),
        "partial_target": partial_target,
        "target": target,
        "image_ids": np.array([str(x) for x in image_ids], dtype=object).reshape(-1, 1),
    }
    savemat(spec.out_dir / "mat_data" / "pll_dataset.mat", mat_payload, do_compression=True)

    # 统计报告
    write_dataset_stats(
        spec,
        pll_df,
        label_order,
        partial_target,
        target,
        extra_metrics={
            "multi_image_rate": multi_image_rate,
            "label_agree_rate": label_agree_rate,
            "partial_jaccard_mean": partial_jaccard_mean,
            "unique_expert_count": int(
                ann["expert_id"].astype(str).str.strip().replace({"nan": ""}).replace({"None": ""}).loc[lambda s: s.ne("")].nunique()
            ),
            "total_record_count": int(len(ann)),
        },
    )


def write_dataset_stats(
    spec: DatasetSpec,
    pll_df: pd.DataFrame,
    label_order: Sequence[int],
    partial_target: np.ndarray,
    target: np.ndarray,
    extra_metrics: Optional[Dict[str, object]] = None,
) -> None:
    q, m = partial_target.shape

    cand_sizes = partial_target.sum(axis=0).astype(int)
    avg_cand = float(np.mean(cand_sizes))
    ambiguity_rate = float(np.mean(cand_sizes > 1))
    partial_ratio = float(np.mean(cand_sizes / max(1, q)))

    # target label distribution
    tgt_counts = target.sum(axis=1).astype(int)
    pt_counts = partial_target.sum(axis=1).astype(int)

    # expert stats
    rec_counts = pll_df["record_count"].astype(int)
    multi_anno_rate = float(np.mean(rec_counts > 1))

    extra_metrics = extra_metrics or {}

    lines: List[str] = []
    lines.append(f"# Dataset {spec.dataset_id} 统计报告")
    lines.append("")
    lines.append("## 基本信息")
    lines.append(f"- 样本数 M: {m}")
    lines.append(f"- 类别数 Q: {q}")
    lines.append(f"- 平均偏标注候选数 E[|S|]: {avg_cand:.4f}")
    lines.append(f"- 偏标注比例 E[|S|/Q]: {partial_ratio:.4f}")
    lines.append(f"- 二义性样本占比 P(|S|>1): {ambiguity_rate:.4f}")
    lines.append(f"- 多次标注样本占比 P(records>1): {multi_anno_rate:.4f}")
    if extra_metrics.get("total_record_count") is not None:
        lines.append(f"- 标注记录总数（annotation records）: {int(extra_metrics.get('total_record_count'))}")
    if extra_metrics.get("unique_expert_count") is not None:
        lines.append(f"- 参与标注的专家数（unique expert_id）: {int(extra_metrics.get('unique_expert_count'))}")

    # 专家一致性（若存在多次标注样本）
    if extra_metrics.get("label_agree_rate") is not None:
        lines.append(f"- 多次标注图片占比 P(images with >1 records): {float(extra_metrics.get('multi_image_rate')):.4f}")
        lines.append(f"- 多次标注 target(label_id) 一致率: {float(extra_metrics.get('label_agree_rate')):.4f}")
    if extra_metrics.get("partial_jaccard_mean") is not None:
        lines.append(f"- 多次标注 partial(label_ids) 平均 Jaccard: {float(extra_metrics.get('partial_jaccard_mean')):.4f}")

    lines.append("")
    lines.append("## 标签分布（target）")
    lines.append("label_id,count")
    for lid, c in zip(label_order, tgt_counts.tolist()):
        lines.append(f"{lid},{int(c)}")

    lines.append("")
    lines.append("## 标签分布（partial_target）")
    lines.append("label_id,count")
    for lid, c in zip(label_order, pt_counts.tolist()):
        lines.append(f"{lid},{int(c)}")

    out = "\n".join(lines).rstrip() + "\n"
    (spec.out_dir / "data_stats.md").write_text(out, encoding="utf-8")


def validate_outputs(repo_root: Path, only_datasets: Optional[Set[int]] = None) -> None:
    """批量校验：映射表无缺失、数据集文件可加载。

    only_datasets:
      - None：校验 dataset14/15/16 全部
      - {14}：仅校验 dataset14（以及 core_mappings）
    """
    # core mappings
    core = repo_root / "core_mappings"
    for f in ["dataset_mapping.csv", "image_id_mapping.csv", "label_dict.csv"]:
        p = core / f
        if not p.exists():
            raise FileNotFoundError(f"缺失核心文件: {p}")
        # 允许空字符串（例如 label_dict.description 暂为空），但不允许真正的 NaN
        df = pd.read_csv(p, keep_default_na=False)
        if df.isna().any().any():
            raise ValueError(f"{p} 存在缺失值")

    dataset_mapping = pd.read_csv(core / "dataset_mapping.csv", keep_default_na=False)
    label_dict = pd.read_csv(core / "label_dict.csv", keep_default_na=False)

    ds_to_q = {int(r["dataset_id"]): int(r["num_classes"]) for _, r in dataset_mapping.iterrows()}
    ds_to_label_ids: Dict[int, Set[int]] = {}
    for ds_id, grp in label_dict.groupby("dataset_id"):
        ds_to_label_ids[int(ds_id)] = set(int(x) for x in grp["label_id"].tolist())

    # per dataset
    ds_dirs = [(14, "dataset14"), (15, "dataset15"), (16, "dataset16")]
    if only_datasets is not None:
        ds_dirs = [(ds_id, dname) for ds_id, dname in ds_dirs if ds_id in only_datasets]

    for _ds_id, d in ds_dirs:
        base = repo_root / d
        for p in [
            base / "csv_data" / "pll_dataset.csv",
            base / "csv_data" / "partial_target.parquet",
            base / "csv_data" / "target.parquet",
            base / "mat_data" / "pll_dataset.mat",
            base / "id_mapping.csv",
            base / "data_stats.md",
        ]:
            if not p.exists():
                raise FileNotFoundError(f"缺失数据集文件: {p}")

        # load check
        pll = pd.read_csv(base / "csv_data" / "pll_dataset.csv", keep_default_na=False)
        if pll.isna().any().any():
            raise ValueError(f"{d} pll_dataset.csv 存在缺失值")
        if pll["image_id"].duplicated().any():
            raise ValueError(f"{d} image_id 非唯一")
        if (pll["image_path"].astype(str).str.contains(r"^[0-9a-f]{32}_", regex=True)).any():
            raise ValueError(f"{d} image_path 仍包含 UUID 前缀")

        dataset_id = int(pll["dataset_id"].iloc[0])
        q_expected = int(ds_to_q.get(dataset_id))
        label_ids = ds_to_label_ids.get(dataset_id, set())
        if not q_expected or len(label_ids) != q_expected:
            raise ValueError(f"{d} label_dict 数量不一致: expected={q_expected} got={len(label_ids)}")

        image_ids: List[int] = [int(x) for x in pll["image_id"].tolist()]

        pt = pd.read_parquet(base / "csv_data" / "partial_target.parquet")
        tg = pd.read_parquet(base / "csv_data" / "target.parquet")

        # partial_target: Q×M
        if pt.shape[0] != q_expected or pt.shape[1] != len(image_ids):
            raise ValueError(f"{d} partial_target 维度错误: {pt.shape} expected=({q_expected},{len(image_ids)})")
        # index should be label_id
        pt_index = [int(x) for x in list(pt.index)]
        if set(pt_index) != set(label_ids):
            raise ValueError(f"{d} partial_target 行索引(label_id)与 label_dict 不一致")
        # columns should match image_ids
        pt_cols = [int(x) for x in list(pt.columns)]
        if set(pt_cols) != set(image_ids):
            raise ValueError(f"{d} partial_target 列(image_id)与 pll_dataset 不一致")
        # values must be 0/1
        pt_vals = set(np.unique(pt.values))
        if not pt_vals.issubset({0, 1}):
            raise ValueError(f"{d} partial_target 存在非 0/1 值: {sorted(pt_vals)[:10]}")

        # target: 1×M，值为 label_id
        if tg.shape[0] != 1 or tg.shape[1] != len(image_ids):
            raise ValueError(f"{d} target 维度错误: {tg.shape} expected=(1,{len(image_ids)})")
        tg_cols = [int(x) for x in list(tg.columns)]
        if set(tg_cols) != set(image_ids):
            raise ValueError(f"{d} target 列(image_id)与 pll_dataset 不一致")
        tg_vals = set(int(x) for x in tg.iloc[0].tolist())
        if not tg_vals.issubset(label_ids):
            bad = sorted(tg_vals - label_ids)[:10]
            raise ValueError(f"{d} target 存在不在 label_dict 的 label_id: {bad}")


def default_specs(repo_root: Path) -> List[DatasetSpec]:
    res_src = Path("/home/droot/medc-img-annotation-app/scripts/res_src")
    return [
        DatasetSpec(
            dataset_id=14,
            name="NIH Chest X-rays (difficult subset)",
            f4dficimg_subdir="dif_nih14",
            expected_num_classes=14,
            xlsx_path=res_src / "医学图像标注数据_数据集14_admin_20260319_175601.xlsx",
            out_dir=repo_root / "dataset14",
        ),
        DatasetSpec(
            dataset_id=15,
            name="ODIR5K (difficult balanced subset)",
            f4dficimg_subdir="dif_orid5k_balanced",
            expected_num_classes=8,
            xlsx_path=res_src / "医学图像标注数据_数据集15_admin_20260319_175606.xlsx",
            out_dir=repo_root / "dataset15",
        ),
        DatasetSpec(
            dataset_id=16,
            name="Brain Tumor MRI (difficult subset)",
            f4dficimg_subdir="dif_MRItumor",
            expected_num_classes=4,
            xlsx_path=res_src / "医学图像标注数据_数据集16_admin_20260319_175608.xlsx",
            out_dir=repo_root / "dataset16",
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--xlsx14", default="")
    ap.add_argument("--xlsx15", default="")
    ap.add_argument("--xlsx16", default="")
    ap.add_argument(
        "--only-datasets",
        default="",
        help="仅处理指定 dataset_id，逗号分隔，例如 '14' 或 '14,16'。不指定则处理 14/15/16 全部。",
    )
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    specs = default_specs(repo_root)

    only_datasets = parse_only_datasets(args.only_datasets)

    # 覆盖 XLSX 路径
    overrides = {14: args.xlsx14, 15: args.xlsx15, 16: args.xlsx16}
    updated = []
    for s in specs:
        ov = overrides.get(s.dataset_id) or ""
        if ov:
            updated.append(DatasetSpec(**{**s.__dict__, "xlsx_path": Path(ov)}))
        else:
            updated.append(s)
    specs = updated

    if args.validate_only and not args.export:
        validate_outputs(repo_root, only_datasets=only_datasets)
        print("OK: validate-only")
        return

    if not args.export:
        raise SystemExit("请指定 --export 或 --validate-only")

    # Step1 core_mappings（支持局部更新）
    if only_datasets is None:
        dataset_mapping, image_id_mapping, label_dict = build_core_mappings(specs)
        (repo_root / "core_mappings" / "dataset_mapping.csv").write_text(dataset_mapping.to_csv(index=False), encoding="utf-8")
        (repo_root / "core_mappings" / "image_id_mapping.csv").write_text(image_id_mapping.to_csv(index=False), encoding="utf-8")
        (repo_root / "core_mappings" / "label_dict.csv").write_text(label_dict.to_csv(index=False), encoding="utf-8")
        specs_to_export = specs
    else:
        specs_to_export = [s for s in specs if s.dataset_id in only_datasets]
        if not specs_to_export:
            raise SystemExit(f"--only-datasets={args.only_datasets} 未匹配到任何数据集（仅支持 14/15/16）")
        update_core_mappings_partial(repo_root, specs_to_export, all_specs=specs)

    # Step2 export datasets (CSV/Parquet/Mat/Stats)
    for spec in specs_to_export:
        export_one_dataset(spec)

    # Step6 validate
    validate_outputs(repo_root, only_datasets=only_datasets)
    print("OK: export + validate")


if __name__ == "__main__":
    main()
