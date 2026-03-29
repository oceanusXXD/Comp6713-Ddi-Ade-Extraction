"""对处理后的 ADE/DDI ChatML 数据集做去重清理。

这个脚本采用保守策略，只处理“重复”相关问题：

1. 把每一行重写成仓库训练阶段使用的规范格式。
2. 移除 train 与 validation/test 的精确文本重叠，以及 validation 与 test 的精确重叠。
3. 将同一个 split 内的完全重复文本压缩为一行。
4. 当重复文本之间的标签集合不一致时，保留多数关系集合。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import extract_training_example, read_jsonl, serialize_relations
from src.observability import write_json
from src.prompting import build_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对处理后的 ChatML train/validation/test 文件做去重。")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean",
        help="包含 merged_chatml_{train,validation,test}.jsonl 的目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="可选输出目录。默认等于 --input-dir，也就是原地重写。",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default=None,
        help="可选统计文件输出路径。默认写到 <output-dir>/dedup_stats.json。",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """做轻量文本标准化，主要用于折叠空白。"""
    return " ".join(str(text).strip().split())


def canonical_text(text: str) -> str:
    """生成去重比对使用的文本键。"""
    return normalize_text(text).casefold()


def build_entry(row: Dict[str, Any], *, line_no: int) -> Dict[str, Any]:
    """把原始行转换成便于后续去重分析的结构。"""
    example = extract_training_example(row)
    canonical_row = {
        "messages": build_messages(
            example["system_prompt"],
            example["user_text"],
            serialize_relations(example["relations"]),
        )
    }
    if "augmentation_type" in row:
        canonical_row["augmentation_type"] = row["augmentation_type"]
    return {
        "line_no": line_no,
        "text_key": canonical_text(example["user_text"]),
        "user_text": example["user_text"],
        "relations": example["relations"],
        "relation_signature": serialize_relations(example["relations"]),
        "relation_count": len(example["relations"]),
        "canonical_row": canonical_row,
    }


def load_entries(path: Path) -> List[Dict[str, Any]]:
    """读取并解析一个 split 的所有样本。"""
    rows = read_jsonl(path)
    return [build_entry(row, line_no=index + 1) for index, row in enumerate(rows)]


def exact_overlap(texts_a: Iterable[str], texts_b: Iterable[str]) -> List[str]:
    """返回两组文本之间的精确重叠集合。"""
    return sorted(set(texts_a) & set(texts_b))


def deduplicate_entries(entries: Sequence[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """对单个 split 内的重复文本做压缩。"""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["text_key"]].append(entry)

    deduped: List[Dict[str, Any]] = []
    duplicate_groups = 0
    duplicate_rows_removed = 0
    conflict_groups = 0
    examples: List[Dict[str, Any]] = []

    for text_key, group in grouped.items():
        if len(group) == 1:
            deduped.append(group[0]["canonical_row"])
            continue

        duplicate_groups += 1
        duplicate_rows_removed += len(group) - 1
        signature_counts = Counter(entry["relation_signature"] for entry in group)
        if len(signature_counts) > 1:
            conflict_groups += 1
        chosen = sorted(
            group,
            key=lambda entry: (
                -signature_counts[entry["relation_signature"]],
                -entry["relation_count"],
                entry["line_no"],
            ),
        )[0]
        deduped.append(chosen["canonical_row"])

        if len(examples) < 10:
            examples.append(
                {
                    "text": group[0]["user_text"][:500],
                    "source_lines": [entry["line_no"] for entry in group],
                    "relation_signature_counts": dict(signature_counts),
                    "chosen_line": chosen["line_no"],
                }
            )

    return deduped, {
        "duplicate_groups_collapsed": duplicate_groups,
        "duplicate_rows_removed": duplicate_rows_removed,
        "conflict_groups": conflict_groups,
        "sample_groups": examples,
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """把结果写成 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_hash(rows_by_split: Dict[str, Sequence[Dict[str, Any]]]) -> str:
    """为当前数据集版本生成稳定哈希。"""
    digest = hashlib.sha256()
    for split in ("train", "validation", "test"):
        for row in rows_by_split[split]:
            digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()[:12]


def count_duplicate_text_rows(rows: Sequence[Dict[str, Any]]) -> int:
    """统计一个 split 内仍然剩余的重复文本行数。"""
    counter = Counter(
        canonical_text(extract_training_example(row)["user_text"])
        for row in rows
    )
    return sum(count - 1 for count in counter.values() if count > 1)


def main() -> None:
    args = parse_args()
    input_dir = (PROJECT_ROOT / args.input_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = (
        (PROJECT_ROOT / args.stats_path).resolve()
        if args.stats_path
        else (output_dir / "dedup_stats.json").resolve()
    )

    split_paths = {
        "train": input_dir / "merged_chatml_train.jsonl",
        "validation": input_dir / "merged_chatml_validation.jsonl",
        "test": input_dir / "merged_chatml_test.jsonl",
    }
    entries = {split: load_entries(path) for split, path in split_paths.items()}

    validation_keys = {entry["text_key"] for entry in entries["validation"]}
    test_keys = {entry["text_key"] for entry in entries["test"]}

    removed_train_overlap = [
        {"line": entry["line_no"], "text": entry["user_text"][:500]}
        for entry in entries["train"]
        if entry["text_key"] in validation_keys or entry["text_key"] in test_keys
    ]
    removed_validation_overlap = [
        {"line": entry["line_no"], "text": entry["user_text"][:500]}
        for entry in entries["validation"]
        if entry["text_key"] in test_keys
    ]

    filtered_entries = {
        "train": [
            entry for entry in entries["train"]
            if entry["text_key"] not in validation_keys and entry["text_key"] not in test_keys
        ],
        "validation": [
            entry for entry in entries["validation"]
            if entry["text_key"] not in test_keys
        ],
        "test": list(entries["test"]),
    }

    deduped_rows: Dict[str, List[Dict[str, Any]]] = {}
    per_split_stats: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        deduped, split_stats = deduplicate_entries(filtered_entries[split])
        deduped_rows[split] = deduped
        per_split_stats[split] = {
            "rows_in": len(entries[split]),
            "rows_after_cross_split_overlap_filter": len(filtered_entries[split]),
            "rows_out": len(deduped),
            "cross_split_overlap_rows_removed": len(entries[split]) - len(filtered_entries[split]),
            **split_stats,
        }

    output_paths = {
        split: output_dir / f"merged_chatml_{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    for split, path in output_paths.items():
        write_jsonl(path, deduped_rows[split])

    overlaps_after = {
        "train_validation_exact_text_overlap": len(
            exact_overlap(
                (extract_training_example(row)["user_text"] for row in deduped_rows["train"]),
                (extract_training_example(row)["user_text"] for row in deduped_rows["validation"]),
            )
        ),
        "train_test_exact_text_overlap": len(
            exact_overlap(
                (extract_training_example(row)["user_text"] for row in deduped_rows["train"]),
                (extract_training_example(row)["user_text"] for row in deduped_rows["test"]),
            )
        ),
        "validation_test_exact_text_overlap": len(
            exact_overlap(
                (extract_training_example(row)["user_text"] for row in deduped_rows["validation"]),
                (extract_training_example(row)["user_text"] for row in deduped_rows["test"]),
            )
        ),
    }

    previous_manifest_path = input_dir / "manifest.json"
    previous_manifest = {}
    if previous_manifest_path.exists():
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    notes = list(previous_manifest.get("notes", []))
    notes.extend(
        [
            "已从 train 中移除与 test 的精确文本重叠",
            "已从 validation 中移除与 test 的精确文本重叠",
            "已将每个 split 内的完全重复文本压缩为一行",
            "当重复文本的标签集合不一致时，保留多数关系集合",
            "已将 assistant 目标重写为规范序列化关系格式",
        ]
    )

    manifest = {
        "dataset_version": output_dir.name,
        "train_path": str(output_paths["train"]),
        "validation_path": str(output_paths["validation"]),
        "test_path": str(output_paths["test"]),
        "dataset_hash": dataset_hash(deduped_rows),
        "notes": notes,
        "dedup_stage": {
            "train_cross_split_removed": len(removed_train_overlap),
            "validation_cross_split_removed": len(removed_validation_overlap),
            "train_duplicate_rows_removed": per_split_stats["train"]["duplicate_rows_removed"],
            "validation_duplicate_rows_removed": per_split_stats["validation"]["duplicate_rows_removed"],
            "test_duplicate_rows_removed": per_split_stats["test"]["duplicate_rows_removed"],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "removed_cross_split_examples": {
            "train": removed_train_overlap[:10],
            "validation": removed_validation_overlap[:10],
        },
        "per_split": per_split_stats,
        "post_dedup_duplicate_text_rows": {
            split: count_duplicate_text_rows(rows)
            for split, rows in deduped_rows.items()
        },
        "post_dedup_overlaps": overlaps_after,
        "dataset_hash": manifest["dataset_hash"],
    }
    write_json(stats_path, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
