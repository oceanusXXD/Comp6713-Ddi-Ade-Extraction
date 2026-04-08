#!/usr/bin/env python3
"""构建评测数据索引文件。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "evaluate_datasets"
REPORT_STATS = REPO_ROOT / "reports" / "augment" / "data_stats.json"
FALLBACK_STATS = REPO_ROOT / "reports" / "data_stats.json"
OUTPUT_JSON = EVAL_ROOT / "DATASET_INDEX.json"
OUTPUT_MD = EVAL_ROOT / "DATASET_INDEX.md"


def ensure_parent(path: Path) -> None:
    """确保输出文件父目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl_rows(path: Path) -> int | None:
    """统计 JSONL 行数；不存在时返回 `None`。"""
    if not path.exists():
        return None
    return sum(1 for _ in path.open("r", encoding="utf-8"))


def build_index() -> Dict[str, Any]:
    """收集内部与外部评测数据，组装统一索引结构。"""
    if REPORT_STATS.exists():
        stats = read_json(REPORT_STATS)
    elif FALLBACK_STATS.exists():
        stats = read_json(FALLBACK_STATS)
    else:
        stats = {}
    manifest_path = EVAL_ROOT / "MANIFEST.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    train_path = REPO_ROOT / "data" / "processed" / "Comp6713-Ddi-Ade-Extraction_final_augment" / "merged_chatml_train.jsonl"
    validation_path = REPO_ROOT / "data" / "processed" / "Comp6713-Ddi-Ade-Extraction_final_augment" / "merged_chatml_validation.jsonl"
    test_path = REPO_ROOT / "data" / "processed" / "Comp6713-Ddi-Ade-Extraction_final_augment" / "merged_chatml_test.jsonl"
    augment_path = REPO_ROOT / "data" / "processed" / "Comp6713-Ddi-Ade-Extraction_final_augment" / "merged_chatml_train_augmentations.jsonl"

    train_rows = count_jsonl_rows(train_path)
    validation_rows = count_jsonl_rows(validation_path)
    test_rows = count_jsonl_rows(test_path)
    augment_rows = count_jsonl_rows(augment_path)
    augment_type_counts = (
        stats.get("files", {})
        .get("final_augmentations", {})
        .get("augmentation_type_counts", {})
    )

    index = {
        "internal": {
            "train": {
                "path": str(train_path),
                "num_rows": stats.get("files", {}).get("final_train", {}).get("num_rows", train_rows),
            },
            "validation": {
                "path": str(validation_path),
                "num_rows": stats.get("files", {}).get("original_validation", {}).get("num_rows", validation_rows),
            },
            "test": {
                "path": str(test_path),
                "num_rows": stats.get("files", {}).get("original_test", {}).get("num_rows", test_rows),
            },
            "augmentations": {
                "path": str(augment_path),
                "num_rows": stats.get("files", {}).get("final_augmentations", {}).get("num_rows", augment_rows),
                "augmentation_type_counts": augment_type_counts,
            },
        },
        "external": {
            "seen_style_core": {
                "readme": str(EVAL_ROOT / "seen_style_core" / "README.md"),
                "validation_path": str(EVAL_ROOT / "seen_style_core" / "official_held_out" / "merged_chatml_validation.jsonl"),
                "test_path": str(EVAL_ROOT / "seen_style_core" / "official_held_out" / "merged_chatml_test.jsonl"),
                "validation_rows": count_jsonl_rows(EVAL_ROOT / "seen_style_core" / "official_held_out" / "merged_chatml_validation.jsonl"),
                "test_rows": count_jsonl_rows(EVAL_ROOT / "seen_style_core" / "official_held_out" / "merged_chatml_test.jsonl"),
                "used_by_default_training": False,
                "used_by_default_validation": False,
                "used_as_optional_same_style_eval": True,
            },
            "bundles": manifest,
        },
    }
    return index


def build_markdown(index: Dict[str, Any]) -> str:
    """把索引结构转换成 Markdown 说明。"""
    internal = index["internal"]
    external = index["external"]
    lines = [
        "# 评测数据索引",
        "",
        "## 当前内部训练 / 验证 / 测试数据",
        "",
        f"- 训练数据：`{internal['train']['path']}`，共 `{internal['train']['num_rows']}` 条",
        f"- 验证数据：`{internal['validation']['path']}`，共 `{internal['validation']['num_rows']}` 条",
        f"- 测试数据：`{internal['test']['path']}`，共 `{internal['test']['num_rows']}` 条",
        f"- 增强 sidecar：`{internal['augmentations']['path']}`，共 `{internal['augmentations']['num_rows']}` 条",
        f"- 增强类型分布：`{json.dumps(internal['augmentations']['augmentation_type_counts'], ensure_ascii=False)}`",
        "",
        "## 当前外部评测数据",
        "",
        f"- 同风格 held-out 验证集：`{external['seen_style_core']['validation_path']}`，共 `{external['seen_style_core']['validation_rows']}` 条",
        f"- 同风格 held-out 测试集：`{external['seen_style_core']['test_path']}`，共 `{external['seen_style_core']['test_rows']}` 条",
        "- `seen_style_core` 当前不是主线默认训练 / 默认验证配置的一部分，只是可选的同风格补充评测入口。",
        "",
        "## 外部 bundle 目录",
        "",
    ]

    bundles = external["bundles"]
    for bundle_name in sorted(bundles):
        bundle = bundles[bundle_name]
        status = bundle.get("status", "unknown")
        lines.append(f"- `{bundle_name}`：状态 `{status}`")

    lines.extend(
        [
            "",
            "## 用法",
            "",
            "- 重新下载并整理外部评测集：`bash evaluate_datasets/download_evaluate_datasets.sh`",
            "- 重新生成索引：`.venv/bin/python evaluate_datasets/build_dataset_index.py`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """写出 JSON 和 Markdown 两份索引文件。"""
    index = build_index()
    ensure_parent(OUTPUT_JSON)
    OUTPUT_JSON.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(index), encoding="utf-8")
    print(f"Saved JSON index to: {OUTPUT_JSON}")
    print(f"Saved Markdown index to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
