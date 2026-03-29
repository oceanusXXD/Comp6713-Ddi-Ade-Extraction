"""对处理后的 ADE/DDI 数据集做非重复类噪声清理。

这一轮清理专门处理仍会影响训练的标签噪声：

1. 删除 head_entity == tail_entity 的自环关系。
2. 将同一实体对的多标签 DDI 冲突收敛为单标签。
3. 如果一行在清理后变成空标签，则删除该行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import extract_training_example, read_jsonl, serialize_relations
from src.observability import write_json
from src.prompting import build_messages

RELATION_PRIORITY = {
    "ADE": 100,
    "DDI-ADVISE": 90,
    "DDI-MECHANISM": 80,
    "DDI-EFFECT": 70,
    "DDI-INT": 60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清理处理后 ChatML train/validation/test 文件中的标签噪声。")
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
        help="可选统计文件输出路径。默认写到 <output-dir>/sanitize_stats.json。",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """做轻量文本标准化，主要用于折叠空白。"""
    return " ".join(str(text).strip().split())


def canonical_text(text: str) -> str:
    """生成文本的规范比较键。"""
    return normalize_text(text).casefold()


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """把结果写成 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def relation_sort_key(relation: Dict[str, str]) -> Tuple[int, str, str, str]:
    """为冲突标签选择优先级排序键。"""
    return (
        -RELATION_PRIORITY.get(relation["relation_type"], 0),
        relation["relation_type"],
        relation["head_entity"].casefold(),
        relation["tail_entity"].casefold(),
    )


def choose_relation(relations: Sequence[Dict[str, str]]) -> Dict[str, str]:
    """从同一实体对的候选关系中选出保留标签。"""
    return sorted(relations, key=relation_sort_key)[0]


def sanitize_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    """清理单行样本中的自环关系和多标签冲突。"""
    example = extract_training_example(row)
    original_relations = example["relations"]

    removed_self_loops: List[Dict[str, str]] = []
    filtered_relations: List[Dict[str, str]] = []
    for relation in original_relations:
        if relation["head_entity"].casefold() == relation["tail_entity"].casefold():
            removed_self_loops.append(relation)
            continue
        filtered_relations.append(relation)

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for relation in filtered_relations:
        key = (relation["head_entity"].casefold(), relation["tail_entity"].casefold())
        grouped[key].append(relation)

    cleaned_relations: List[Dict[str, str]] = []
    resolved_conflicts: List[Dict[str, Any]] = []
    for group in grouped.values():
        label_set = {relation["relation_type"] for relation in group}
        if len(label_set) > 1:
            chosen = choose_relation(group)
            resolved_conflicts.append(
                {
                    "pair": [chosen["head_entity"], chosen["tail_entity"]],
                    "labels_before": sorted(label_set),
                    "label_after": chosen["relation_type"],
                }
            )
            cleaned_relations.append(chosen)
        else:
            cleaned_relations.append(group[0])

    if not cleaned_relations:
        return None, {
            "removed_self_loops": removed_self_loops,
            "resolved_conflicts": resolved_conflicts,
            "dropped_empty": True,
            "user_text": example["user_text"],
        }

    sanitized_row = {
        "messages": build_messages(
            example["system_prompt"],
            example["user_text"],
            serialize_relations(cleaned_relations),
        )
    }
    if "augmentation_type" in row:
        sanitized_row["augmentation_type"] = row["augmentation_type"]

    return sanitized_row, {
        "removed_self_loops": removed_self_loops,
        "resolved_conflicts": resolved_conflicts,
        "dropped_empty": False,
        "user_text": example["user_text"],
    }


def dataset_hash(rows_by_split: Dict[str, Sequence[Dict[str, Any]]]) -> str:
    """为当前数据集版本生成稳定哈希。"""
    digest = hashlib.sha256()
    for split in ("train", "validation", "test"):
        for row in rows_by_split[split]:
            digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()[:12]


def remaining_issue_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """统计清理完成后剩余问题数量。"""
    self_loops = 0
    multi_label_pairs = 0
    for row in rows:
        example = extract_training_example(row)
        pair_labels: Dict[Tuple[str, str], set[str]] = defaultdict(set)
        for relation in example["relations"]:
            if relation["head_entity"].casefold() == relation["tail_entity"].casefold():
                self_loops += 1
            pair_labels[(relation["head_entity"].casefold(), relation["tail_entity"].casefold())].add(
                relation["relation_type"]
            )
        multi_label_pairs += sum(1 for labels in pair_labels.values() if len(labels) > 1)
    return {
        "self_loop_relations": self_loops,
        "multi_label_pairs": multi_label_pairs,
    }


def main() -> None:
    args = parse_args()
    input_dir = (PROJECT_ROOT / args.input_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = (
        (PROJECT_ROOT / args.stats_path).resolve()
        if args.stats_path
        else (output_dir / "sanitize_stats.json").resolve()
    )

    split_paths = {
        "train": input_dir / "merged_chatml_train.jsonl",
        "validation": input_dir / "merged_chatml_validation.jsonl",
        "test": input_dir / "merged_chatml_test.jsonl",
    }

    sanitized_rows: Dict[str, List[Dict[str, Any]]] = {}
    per_split_stats: Dict[str, Dict[str, Any]] = {}

    for split, path in split_paths.items():
        rows = read_jsonl(path)
        output_rows: List[Dict[str, Any]] = []
        removed_self_loop_count = 0
        resolved_conflict_pairs = 0
        dropped_empty_rows = 0
        conflict_pattern_counts: Counter[str] = Counter()
        self_loop_examples: List[Dict[str, Any]] = []
        conflict_examples: List[Dict[str, Any]] = []

        for line_no, row in enumerate(rows, 1):
            sanitized, details = sanitize_row(row)
            removed_self_loop_count += len(details["removed_self_loops"])
            resolved_conflict_pairs += len(details["resolved_conflicts"])
            dropped_empty_rows += int(details["dropped_empty"])

            if details["removed_self_loops"] and len(self_loop_examples) < 10:
                self_loop_examples.append(
                    {
                        "line": line_no,
                        "text": details["user_text"][:500],
                        "removed_relations": details["removed_self_loops"],
                    }
                )

            if details["resolved_conflicts"]:
                for conflict in details["resolved_conflicts"]:
                    conflict_pattern_counts[" + ".join(conflict["labels_before"])] += 1
                if len(conflict_examples) < 10:
                    conflict_examples.append(
                        {
                            "line": line_no,
                            "text": details["user_text"][:500],
                            "resolved_pairs": details["resolved_conflicts"],
                        }
                    )

            if sanitized is not None:
                output_rows.append(sanitized)

        sanitized_rows[split] = output_rows
        per_split_stats[split] = {
            "rows_in": len(rows),
            "rows_out": len(output_rows),
            "rows_dropped_empty_after_cleanup": dropped_empty_rows,
            "self_loop_relations_removed": removed_self_loop_count,
            "same_pair_conflicts_resolved": resolved_conflict_pairs,
            "conflict_pattern_counts": dict(conflict_pattern_counts),
            "sample_self_loop_rows": self_loop_examples,
            "sample_conflict_rows": conflict_examples,
        }

    output_paths = {
        split: output_dir / f"merged_chatml_{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    for split, path in output_paths.items():
        write_jsonl(path, sanitized_rows[split])

    manifest_path = output_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    notes = list(manifest.get("notes", []))
    notes.extend(
        [
            "已移除 head==tail 的自环关系",
            "已将同一实体对的多标签冲突收敛为单标签",
            "冲突消解使用 DDI 优先级：DDI-ADVISE > DDI-MECHANISM > DDI-EFFECT > DDI-INT",
            "若一行在自环与冲突清理后变为空标签，则删除该行",
        ]
    )

    dataset_hash_value = dataset_hash(sanitized_rows)
    updated_manifest = {
        **manifest,
        "dataset_version": output_dir.name,
        "train_path": str(output_paths["train"]),
        "validation_path": str(output_paths["validation"]),
        "test_path": str(output_paths["test"]),
        "dataset_hash": dataset_hash_value,
        "notes": notes,
        "sanitize_stage": {
            split: {
                "self_loop_relations_removed": per_split_stats[split]["self_loop_relations_removed"],
                "same_pair_conflicts_resolved": per_split_stats[split]["same_pair_conflicts_resolved"],
                "rows_dropped_empty_after_cleanup": per_split_stats[split]["rows_dropped_empty_after_cleanup"],
            }
            for split in ("train", "validation", "test")
        },
    }
    manifest_path.write_text(json.dumps(updated_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "per_split": per_split_stats,
        "remaining_issues": {
            split: remaining_issue_counts(rows)
            for split, rows in sanitized_rows.items()
        },
        "dataset_hash": dataset_hash_value,
    }
    write_json(stats_path, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
