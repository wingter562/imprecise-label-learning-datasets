#!/usr/bin/env python3
"""数据集标准化导出自查脚本（Agent 执行版）。

- 严格按用户提供的自查清单做验证
- 将执行日志/结果写入 audit_logs/
- 生成 audit_report.md 汇总“通过/不通过”与修正说明

运行：
  python audit_selfcheck.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parent
AUDIT_DIR = REPO_ROOT / "audit_logs"


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_log(filename: str, content: str) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / filename).write_text(content, encoding="utf-8")


def _run_cmd(cmd: Sequence[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def check_core_mappings() -> CheckResult:
    core = REPO_ROOT / "core_mappings"
    required = ["dataset_mapping.csv", "image_id_mapping.csv", "label_dict.csv"]
    missing = [f for f in required if not (core / f).exists()]
    if missing:
        return CheckResult("核心映射表完整性", False, f"缺失文件: {missing}")

    # 非空 + 无 NaN
    for f in required:
        df = _read_csv(core / f)
        if df.empty:
            return CheckResult("核心映射表完整性", False, f"空文件: {f}")
        if df.isna().any().any():
            return CheckResult("核心映射表完整性", False, f"存在 NaN: {f}")

    dm = _read_csv(core / "dataset_mapping.csv")
    exp = {
        14: ("dif_nih14", 14),
        15: ("dif_orid5k_balanced", 8),
        16: ("dif_MRItumor", 4),
    }
    got = {int(r.dataset_id): (r.f4dficimg_subdir, int(r.num_classes)) for r in dm.itertuples(index=False)}
    for k, v in exp.items():
        if k not in got or got[k] != v:
            return CheckResult("核心映射表完整性", False, f"dataset_mapping.csv 映射不匹配: expected {exp} got {got}")

    return CheckResult("核心映射表完整性", True, "core_mappings 三表存在、可读、非空、无 NaN，且 dataset_id 映射符合要求")


def check_structure() -> CheckResult:
    required_root = ["README.md", "data_links.md", "LICENSE", "batch_processor.py"]
    missing = [f for f in required_root if not (REPO_ROOT / f).exists()]
    if missing:
        return CheckResult("目录结构与文档", False, f"根目录缺失: {missing}")

    for ds in [14, 15, 16]:
        base = REPO_ROOT / f"dataset{ds}"
        needed = [base / "csv_data", base / "mat_data", base / "code", base / "id_mapping.csv", base / "data_stats.md"]
        missing_ds = [str(p) for p in needed if not p.exists()]
        if missing_ds:
            return CheckResult("目录结构与文档", False, f"dataset{ds} 缺失: {missing_ds}")

        for f in [base / "csv_data" / "pll_dataset.csv", base / "csv_data" / "partial_target.parquet", base / "csv_data" / "target.parquet", base / "mat_data" / "pll_dataset.mat"]:
            if f.stat().st_size <= 0:
                return CheckResult("目录结构与文档", False, f"文件大小为0: {f}")

        for f in [base / "code" / "data_loader.py", base / "code" / "stats_visualization.py"]:
            if not f.exists() or f.stat().st_size <= 0:
                return CheckResult("目录结构与文档", False, f"缺失/空工具文件: {f}")

    return CheckResult("目录结构与文档", True, "根目录与 dataset14/15/16 结构齐全，核心文件均非空")


def check_csv_parquet_contents(ds: int) -> CheckResult:
    base = REPO_ROOT / f"dataset{ds}"
    pll = _read_csv(base / "csv_data" / "pll_dataset.csv")

    must_cols = {"image_id", "image_path", "dataset_id", "partial_target", "target"}
    if not must_cols.issubset(set(pll.columns)):
        return CheckResult(f"dataset{ds} CSV/Parquet 内容", False, f"pll_dataset.csv 缺失列: {sorted(list(must_cols - set(pll.columns)))}")

    if pll["image_id"].duplicated().any():
        return CheckResult(f"dataset{ds} CSV/Parquet 内容", False, "image_id 存在重复")

    # image_path 去 UUID
    if pll["image_path"].astype(str).str.contains(r"^[0-9a-f]{32}_", regex=True).any():
        return CheckResult(f"dataset{ds} CSV/Parquet 内容", False, "image_path 仍包含 UUID 前缀")

    # partial_target 是可解析的列表/JSON
    sample = pll.head(10)
    try:
        for s in sample["partial_target"].tolist():
            _ = json.loads(s) if isinstance(s, str) and s else []
    except Exception as e:
        return CheckResult(f"dataset{ds} CSV/Parquet 内容", False, f"partial_target 不是合法 JSON list: {e}")

    # Parquet 维度与取值
    pt = pd.read_parquet(base / "csv_data" / "partial_target.parquet")
    tg = pd.read_parquet(base / "csv_data" / "target.parquet")
    q = int((REPO_ROOT / "core_mappings" / "dataset_mapping.csv").read_text(encoding="utf-8").count("\n"))
    # 以 label_dict 为准
    label_ids = set(_read_csv(REPO_ROOT / "core_mappings" / "label_dict.csv").query("dataset_id == @ds")["label_id"].astype(int).tolist())
    q_expected = len(label_ids)
    m = len(pll)

    if pt.shape != (q_expected, m):
        return CheckResult(f"dataset{ds} CSV/Parquet 内容", False, f"partial_target.parquet 维度错误: {pt.shape} expected=({q_expected},{m})")
    vals = set(np.unique(pt.values))
    if not vals.issubset({0, 1}):
        return CheckResult(f"dataset{ds} CSV/Parquet 内容", False, f"partial_target.parquet 存在非0/1: {sorted(list(vals))[:10]}")

    if tg.shape != (1, m):
        return CheckResult(f"dataset{ds} CSV/Parquet 内容", False, f"target.parquet 维度错误: {tg.shape} expected=(1,{m})")

    tg_vals = set(int(x) for x in tg.iloc[0].tolist())
    if not tg_vals.issubset(label_ids):
        bad = sorted(list(tg_vals - label_ids))[:10]
        return CheckResult(f"dataset{ds} CSV/Parquet 内容", False, f"target.parquet 存在不在 label_dict 的 label_id: {bad}")

    if pll.isna().any().any():
        return CheckResult(f"dataset{ds} CSV/Parquet 内容", False, "pll_dataset.csv 存在空值")

    return CheckResult(f"dataset{ds} CSV/Parquet 内容", True, "pll_dataset.csv 字段齐全，Parquet 维度/取值/label_id 范围均正确")


def check_provenance(ds: int) -> CheckResult:
    base = REPO_ROOT / f"dataset{ds}"
    idm = _read_csv(base / "id_mapping.csv")

    need = {"image_id", "uuid_prefix", "original_filename", "f4dficimg_path"}
    if not need.issubset(set(idm.columns)):
        return CheckResult(f"dataset{ds} 溯源映射", False, f"id_mapping.csv 缺失列: {sorted(list(need - set(idm.columns)))}")

    # 抽样验证 uuid_prefix + '_' + original_filename 与 xlsx_image_path 一致（若该列存在）
    if "xlsx_image_path" in idm.columns:
        samp = idm.head(20)
        for r in samp.itertuples(index=False):
            if str(r.uuid_prefix) and str(r.original_filename):
                expect = f"static/img/{r.uuid_prefix}_{r.original_filename}"
                if str(getattr(r, "xlsx_image_path")) != expect:
                    return CheckResult(f"dataset{ds} 溯源映射", False, f"xlsx_image_path 不一致: {getattr(r,'xlsx_image_path')} != {expect}")

    # 100% 文件存在
    missing = []
    for p in idm["f4dficimg_path"].astype(str).tolist():
        physical = Path(p).expanduser()
        if not physical.exists():
            missing.append(str(physical))
            if len(missing) >= 10:
                break
    if missing:
        return CheckResult(f"dataset{ds} 溯源映射", False, f"存在找不到原图路径（示例前10）: {missing}")

    return CheckResult(f"dataset{ds} 溯源映射", True, "id_mapping 字段齐全，且 f4dficimg_path 100% 存在")


def check_mat(ds: int) -> CheckResult:
    base = REPO_ROOT / f"dataset{ds}"
    mat_path = base / "mat_data" / "pll_dataset.mat"
    m = loadmat(mat_path)
    for k in ["data", "partial_target", "target", "image_ids"]:
        if k not in m:
            return CheckResult(f"dataset{ds} Mat 兼容性", False, f"缺失字段: {k}")

    data = m["data"]
    pt = m["partial_target"]
    tgt = m["target"]

    if data.ndim != 2 or data.shape[1] != 1:
        return CheckResult(f"dataset{ds} Mat 兼容性", False, f"data 维度异常: {data.shape}")
    if pt.ndim != 2:
        return CheckResult(f"dataset{ds} Mat 兼容性", False, f"partial_target 维度异常: {pt.shape}")
    if tgt.ndim != 2:
        return CheckResult(f"dataset{ds} Mat 兼容性", False, f"target 维度异常: {tgt.shape}")

    # 不应包含 trainIndex/testIndex
    if "trainIndex" in m or "testIndex" in m:
        return CheckResult(f"dataset{ds} Mat 兼容性", False, "不应包含 trainIndex/testIndex")

    return CheckResult(f"dataset{ds} Mat 兼容性", True, "Mat 字段齐全、维度合理，且不含 train/test 划分字段")


def check_encoding() -> CheckResult:
    # file -I 依赖系统命令；失败则降级
    pll = REPO_ROOT / "dataset14" / "csv_data" / "pll_dataset.csv"
    if not pll.exists():
        return CheckResult("CSV 编码/格式", False, "未找到示例 CSV")

    # 兼容不同 file(1) 版本：优先 --mime，其次 -bi
    for args, log_prefix in [(["file", "--mime", str(pll)], "file_mime"), (["file", "-bi", str(pll)], "file_bi")]:
        code, out = _run_cmd(args)
        _write_log(f"{log_prefix}_{_now_tag()}.log", out)
        if code == 0:
            lower = out.lower()
            if "charset=utf-8" in lower or "utf-8" in lower:
                return CheckResult("CSV 编码/格式", True, out.strip())
            # us-ascii 是 UTF-8 的真子集：只要能被 UTF-8 解码即可视为满足 UTF-8 兼容
            if "charset=us-ascii" in lower or "us-ascii" in lower:
                try:
                    _ = pll.read_bytes().decode("utf-8")
                except Exception as e:
                    return CheckResult("CSV 编码/格式", False, f"file 显示 us-ascii，但 UTF-8 解码失败: {e}")
                return CheckResult("CSV 编码/格式", True, out.strip() + "（us-ascii 为 UTF-8 子集，UTF-8 解码通过）")

            return CheckResult("CSV 编码/格式", False, f"编码未显示为 utf-8: {out.strip()}")

    # 最后降级：用 Python 读取并验证能否以 UTF-8 解码/重编码
    try:
        raw = pll.read_bytes()
        _ = raw.decode("utf-8")
    except Exception as e:
        return CheckResult("CSV 编码/格式", False, f"Python UTF-8 解码失败: {e}")

    return CheckResult("CSV 编码/格式", True, "Python UTF-8 解码通过（系统 file 命令不支持 --mime/-bi 或执行失败）")


def check_traceability_docs() -> CheckResult:
    # data_stats.md 需包含核心统计段（简易关键字检测；允许中英任一）
    missing = []
    for ds in [14, 15, 16]:
        txt = (REPO_ROOT / f"dataset{ds}" / "data_stats.md").read_text(encoding="utf-8")
        has_samples = ("样本数" in txt) or ("sample" in txt.lower()) or ("#images" in txt.lower())
        has_label_dist = ("标签分布" in txt) or ("label distribution" in txt.lower())
        if not (has_samples and has_label_dist):
            missing.append(f"dataset{ds}: 缺少核心统计段（样本数/标签分布）")
    if missing:
        return CheckResult("可追溯性文档", False, "; ".join(missing))

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    readme_l = readme.lower()

    # README 需覆盖：UUID 前缀剥离规则 + static/img 模式 + 上游链路脚本引用
    missing_readme = []

    if "uuid" not in readme_l:
        missing_readme.append("UUID")
    if "static/img" not in readme_l:
        missing_readme.append("static/img")
    if ("strip" not in readme_l) and ("remove" not in readme_l) and ("去掉" not in readme):
        missing_readme.append("strip/remove UUID prefix")
    if "ml4img" not in readme_l:
        missing_readme.append("ml4img")
    if "medc-img-annotation-app" not in readme_l:
        missing_readme.append("medc-img-annotation-app")
    if "aggregate_difficult_multilabel.py" not in readme_l:
        missing_readme.append("aggregate_difficult_multilabel.py")
    if "extract_difficult_images.py" not in readme_l:
        missing_readme.append("extract_difficult_images.py")

    if missing_readme:
        return CheckResult("可追溯性文档", False, f"README.md 缺少关键说明: {missing_readme}")

    # train/test 说明：允许中/英表述
    ok_split_phrases = [
        "未提供 train/test",
        "不提供 train/test",
        "no train/val/test split",
        "no train/test split",
        "no split is provided",
    ]
    if not any(p.lower() in readme_l for p in ok_split_phrases):
        return CheckResult(
            "可追溯性文档",
            False,
            "README.md 缺少不划分 train/val/test 的说明（需包含类似：No train/val/test split is provided / 未提供 train/test）",
        )

    return CheckResult("可追溯性文档", True, "README 与 data_stats.md 包含核心来源/方法/不划分说明")


def main() -> None:
    tag = _now_tag()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    results: List[CheckResult] = []

    # 基础文件完整性
    results.append(check_core_mappings())
    results.append(check_structure())

    # 内容正确性 + 溯源 + Mat
    for ds in [14, 15, 16]:
        results.append(check_csv_parquet_contents(ds))
        results.append(check_provenance(ds))
        results.append(check_mat(ds))

    # 编码/格式
    results.append(check_encoding())

    # 可追溯性
    results.append(check_traceability_docs())

    passed = sum(1 for r in results if r.passed)
    failed = [r for r in results if not r.passed]

    # 写报告
    lines: List[str] = []
    lines.append("# 标准化导出自查报告")
    lines.append("")
    lines.append(f"- 时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 通过项数: {passed}/{len(results)}")
    lines.append("")

    lines.append("## 检查结果")
    for r in results:
        status = "通过" if r.passed else "不通过"
        lines.append(f"- {r.name}: {status} — {r.details}")

    if failed:
        lines.append("")
        lines.append("## 不通过项与修正建议")
        for r in failed:
            lines.append(f"- {r.name}: {r.details}")

    # 核心统计摘要
    lines.append("")
    lines.append("## 核心统计摘要")
    for ds in [14, 15, 16]:
        pll = _read_csv(REPO_ROOT / f"dataset{ds}" / "csv_data" / "pll_dataset.csv")
        label_cnt = len(_read_csv(REPO_ROOT / "core_mappings" / "label_dict.csv").query("dataset_id == @ds"))
        lines.append(f"- dataset{ds}: M={len(pll)} Q={label_cnt}")

    out_text = "\n".join(lines).rstrip() + "\n"
    (REPO_ROOT / "audit_report.md").write_text(out_text, encoding="utf-8")

    # 同步写一份 raw JSON 便于机器处理
    raw = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "total": len(results),
        "results": [{"name": r.name, "passed": r.passed, "details": r.details} for r in results],
    }
    _write_log(f"audit_results_{tag}.json", json.dumps(raw, ensure_ascii=False, indent=2))

    # stdout
    print(out_text)

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
