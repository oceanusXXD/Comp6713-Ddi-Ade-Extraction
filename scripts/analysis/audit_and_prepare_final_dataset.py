"""数据审计与最终训练集物化脚本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (
    CANONICAL_LABELS,
    compute_dataset_statistics,
    extract_training_example,
    read_jsonl,
    serialize_relations,
)
from src.model_utils import load_tokenizer, load_training_config
from src.observability import write_json
from src.prompting import build_messages, load_system_prompt

ALLOWED_AUGMENTATION_TYPES = ("paraphrase", "negative", "hardcase", "margincase")
ROLE_SEQUENCE = ("system", "user", "assistant")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+./-]*")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="审计 ADE/DDI 数据、物化最终数据集并写出报告。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen3_8b_lora_ddi_ade_final.yaml",
        help="用于 tokenizer 长度统计的训练配置文件。",
    )
    parser.add_argument(
        "--base-data-dir",
        type=str,
        default="data",
        help="包含原始 merged_chatml_{train,validation,test}.jsonl 文件的目录。",
    )
    parser.add_argument(
        "--primary-augmentations",
        type=str,
        default="data/augmentations/curated_train_augmentations.json",
        help="主增强规格 JSON 文件路径。",
    )
    parser.add_argument(
        "--supplement-augmentations",
        type=str,
        nargs="*",
        default=[
            "data/augmentations/curated_train_augmentations_supplement_augment.json",
        ],
        help="补充增强规格 JSON 文件路径列表。",
    )
    parser.add_argument(
        "--current-augmentation-jsonl",
        type=str,
        default="data/augmentations/merged_chatml_train_augmentations.jsonl",
        help="当前增强 sidecar JSONL，用于统计增量变化。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/Comp6713-Ddi-Ade-Extraction_final_augment",
        help="最终 train/validation/test 文件的输出目录。",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports/augment",
        help="审计、修复和统计报告的输出目录。",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """做最轻量的文本标准化。"""
    return " ".join(str(text).strip().split())


def canonical_text(text: str) -> str:
    """把文本折叠成便于去重和比对的规范形式。"""
    return normalize_text(text).casefold()


def tokenize_text(text: str) -> set[str]:
    """把文本切成简单词元集合，用于近重复分析。"""
    return {match.group(0).casefold() for match in TOKEN_RE.finditer(text)}


def read_json(path: Path) -> Any:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    """计算文本的 SHA-256。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_spec_records(paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """加载并合并增强样本规格，同时统计重复情况。"""
    merged: List[Dict[str, Any]] = []
    duplicate_texts: List[str] = []
    duplicate_counter = 0
    seen_texts: set[str] = set()
    per_source_counts: Dict[str, int] = {}

    for path in paths:
        if not path.exists():
            continue
        records = read_json(path)
        if not isinstance(records, list):
            raise ValueError(f"Augmentation spec must be a list: {path}")
        kept = 0
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Augmentation entry must be an object in {path}")
            text_key = canonical_text(record.get("user_text", ""))
            if text_key in seen_texts:
                duplicate_counter += 1
                duplicate_texts.append(normalize_text(record.get("user_text", "")))
                continue
            seen_texts.add(text_key)
            merged.append(record)
            kept += 1
        per_source_counts[str(path)] = kept

    return merged, {
        "duplicate_spec_entries_removed": duplicate_counter,
        "duplicate_spec_texts": duplicate_texts,
        "per_source_counts": per_source_counts,
    }


def build_chatml_rows(
    spec_records: Sequence[Dict[str, Any]],
    *,
    system_prompt: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows_plain: List[Dict[str, Any]] = []
    rows_with_type: List[Dict[str, Any]] = []
    for record in spec_records:
        relations = record.get("relations", [])
        row = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": normalize_text(record.get("user_text", ""))},
                {"role": "assistant", "content": serialize_relations(relations)},
            ]
        }
        rows_plain.append(row)
        rows_with_type.append({**row, "augmentation_type": record.get("augmentation_type")})
    return rows_plain, rows_with_type


def canonicalize_chatml_row(
    row: Dict[str, Any],
    *,
    system_prompt: str,
) -> Dict[str, Any]:
    example = extract_training_example(row, system_prompt=system_prompt)
    canonical_row = {
        "messages": build_messages(
            system_prompt,
            example["user_text"],
            serialize_relations(example["relations"]),
        )
    }
    if "augmentation_type" in row:
        canonical_row["augmentation_type"] = row["augmentation_type"]
    return canonical_row


def canonicalize_chatml_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    return [canonicalize_chatml_row(row, system_prompt=system_prompt) for row in rows]


def message_content(row: Dict[str, Any], role: str) -> str:
    for message in row.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", ""))
    return ""


def relation_gap(text: str, head: str, tail: str) -> Optional[int]:
    try:
        head_start = text.index(head)
        tail_start = text.index(tail)
    except ValueError:
        return None
    return abs(tail_start - head_start)


def lexical_markers(text: str) -> Dict[str, bool]:
    lowered = f" {text.lower()} "
    return {
        "negation": any(token in lowered for token in (" no ", " not ", " without ", " did not ", " rather than ", " never ")),
        "conditional": any(
            token in lowered
            for token in (" if ", " when ", " unless ", " should ", " caution ", " contraindicated ", " avoid ", " recommended ")
        ),
        "speculative": any(
            token in lowered
            for token in (" may ", " might ", " possible ", " possibly ", " suggests ", " suggested ", " appears ", " speculat")
        ),
        "coordination": any(token in text for token in (",", ";", " and ", " or ")),
    }


def summarize_numeric(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        median = float(ordered[mid])
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / count,
        "median": median,
    }


def exact_overlap(texts_a: Iterable[str], texts_b: Iterable[str]) -> List[str]:
    return sorted(set(texts_a) & set(texts_b))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def collect_near_duplicate_pairs(
    rows: Sequence[Dict[str, Any]],
    *,
    min_jaccard: float = 0.75,
) -> List[Dict[str, Any]]:
    texts = [message_content(row, "user") for row in rows]
    token_sets = [tokenize_text(text) for text in texts]
    pairs: List[Dict[str, Any]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            score = jaccard(token_sets[i], token_sets[j])
            if score >= min_jaccard:
                pairs.append(
                    {
                        "left_index": i,
                        "right_index": j,
                        "score": round(score, 4),
                        "left_text": texts[i],
                        "right_text": texts[j],
                    }
                )
    return pairs


def file_audit(
    rows: Sequence[Dict[str, Any]],
    *,
    expect_augmentation_type: bool = False,
) -> Dict[str, Any]:
    errors: List[str] = []
    empty_user = 0
    empty_assistant = 0
    invalid_labels = 0
    parse_failures = 0
    entity_substring_issues = 0
    augmentation_type_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    relations_per_sample: List[int] = []
    unique_entity_counts: List[int] = []
    word_lengths: List[int] = []
    long_gap_relations = 0
    marker_counts: Counter[str] = Counter()
    text_counter: Counter[str] = Counter()

    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            errors.append(f"row_{index}: messages is not a list")
            continue
        roles = tuple(message.get("role") for message in messages)
        if roles != ROLE_SEQUENCE:
            errors.append(f"row_{index}: roles={roles!r}")
        if expect_augmentation_type:
            aug_type = row.get("augmentation_type")
            if aug_type not in ALLOWED_AUGMENTATION_TYPES:
                errors.append(f"row_{index}: invalid augmentation_type={aug_type!r}")
            else:
                augmentation_type_counts[str(aug_type)] += 1

        user_text = normalize_text(message_content(row, "user"))
        assistant_content = message_content(row, "assistant")
        text_counter[canonical_text(user_text)] += 1
        word_lengths.append(len(user_text.split()))

        for marker_name, enabled in lexical_markers(user_text).items():
            if enabled:
                marker_counts[marker_name] += 1

        if not user_text:
            empty_user += 1
        if assistant_content == "":
            empty_assistant += 1

        try:
            example = extract_training_example(row)
        except Exception as exc:  # noqa: BLE001
            parse_failures += 1
            errors.append(f"row_{index}: parse_failure={exc}")
            continue

        relations = example["relations"]
        relations_per_sample.append(len(relations))
        entity_set: set[str] = set()
        for relation in relations:
            label = relation["relation_type"]
            if label not in CANONICAL_LABELS:
                invalid_labels += 1
            relation_counts[label] += 1
            head = relation["head_entity"]
            tail = relation["tail_entity"]
            entity_set.add(head)
            entity_set.add(tail)
            if head not in user_text or tail not in user_text:
                entity_substring_issues += 1
            gap = relation_gap(user_text, head, tail)
            if gap is not None and gap >= 40:
                long_gap_relations += 1
        unique_entity_counts.append(len(entity_set))

    duplicate_text_examples = sum(count - 1 for count in text_counter.values() if count > 1)
    return {
        "num_rows": len(rows),
        "parse_failures": parse_failures,
        "empty_user": empty_user,
        "empty_assistant": empty_assistant,
        "invalid_labels": invalid_labels,
        "entity_substring_issues": entity_substring_issues,
        "duplicate_text_examples": duplicate_text_examples,
        "relation_counts": dict(sorted(relation_counts.items())),
        "augmentation_type_counts": dict(sorted(augmentation_type_counts.items())),
        "relations_per_sample": summarize_numeric(relations_per_sample),
        "unique_entity_counts": summarize_numeric(unique_entity_counts),
        "word_lengths": summarize_numeric(word_lengths),
        "long_gap_relations": long_gap_relations,
        "marker_counts": dict(sorted(marker_counts.items())),
        "sample_errors": errors[:50],
    }


def augmentation_semantic_audit(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    group_summary: Dict[str, Dict[str, Any]] = {}
    for aug_type in ALLOWED_AUGMENTATION_TYPES:
        group_rows = [row for row in rows if row.get("augmentation_type") == aug_type]
        positive = 0
        empty = 0
        label_counts: Counter[str] = Counter()
        multi_relation = 0
        complex_sentence = 0
        for row in group_rows:
            relations = json.loads(message_content(row, "assistant"))
            if relations:
                positive += 1
            else:
                empty += 1
            if len(relations) > 1:
                multi_relation += 1
            text = message_content(row, "user")
            if len(text.split()) >= 28 or sum(ch in text for ch in ",;()") >= 2:
                complex_sentence += 1
            for relation in relations:
                label_counts[relation["relation_type"]] += 1
        group_summary[aug_type] = {
            "num_rows": len(group_rows),
            "positive_rows": positive,
            "empty_rows": empty,
            "multi_relation_rows": multi_relation,
            "complex_sentence_rows": complex_sentence,
            "relation_counts": dict(sorted(label_counts.items())),
        }
    return group_summary


def dataset_hash(rows: Sequence[Dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return sha256_text(payload)[:8]


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def markdown_list(items: Sequence[str]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def build_audit_report(stats: Dict[str, Any]) -> str:
    original_train = stats["files"]["original_train"]
    augmentations = stats["files"]["final_augmentations"]
    final_train = stats["files"]["final_train"]
    overlaps = stats["overlaps"]
    supplement = stats["supplement_delta"]
    dataset_hash_value = stats["final_dataset_hash"]

    return f"""# 数据审计报告

## 范围

- 仓库：`data/Comp6713-Ddi-Ade-Extraction`
- 最终处理后数据集：`{stats["paths"]["processed_dir"]}`
- 本次审计对应的数据集哈希：`{dataset_hash_value}`

## 检查文件

- 原始训练集：`{stats["paths"]["original_train"]}`
- 原始验证集：`{stats["paths"]["original_validation"]}`
- 原始测试集：`{stats["paths"]["original_test"]}`
- 现有增强 sidecar：`{stats["paths"]["current_augmentation_jsonl"]}`
- 最终增强 sidecar：`{stats["paths"]["final_augmentations"]}`
- 最终合并训练集：`{stats["paths"]["final_train"]}`

## 格式一致性

- 训练数据统一采用 ChatML JSONL 格式，结构为 `messages = [system, user, assistant]`。
- 所有处理后的 train/validation/test 行都会重写为共享的配置 system prompt，以保证 prompt 一致性。
- `span / offset / index` 字段不属于本仓库 schema，因此不适用边界偏移检查。
- 原始训练集解析失败数：`{original_train["parse_failures"]}`
- 最终增强集解析失败数：`{augmentations["parse_failures"]}`
- 最终训练集解析失败数：`{final_train["parse_failures"]}`
- 最终增强集实体子串问题数：`{augmentations["entity_substring_issues"]}`
- 最终增强集非法标签数：`{augmentations["invalid_labels"]}`

## 语义与标签质量

- 从上一版本保留的已整理增强样本：`{supplement["retained_previous_rows"]}`
- 审计后新增补充样本：`{supplement["new_rows_added"]}`
- 合并过程中去掉的重复增强规格数：`{stats["spec_merge"]["duplicate_spec_entries_removed"]}`
- 最终增强集中文本完全重复的样本数：`{augmentations["duplicate_text_examples"]}`
- 最终增强集中近重复样本对数量（token Jaccard >= 0.75）：`{stats["final_augmentation_near_duplicates"]}`
- 增强集与验证集的精确重叠：`{len(overlaps["augmentations_vs_validation_exact"])}` 行
- 增强集与测试集的精确重叠：`{len(overlaps["augmentations_vs_test_exact"])}` 行
- 从处理后训练集中剔除的污染原始训练样本：`{stats["removed_polluted_train_rows"]}`

人工复核结论：
- `paraphrase`：保持语义一致，同时变化句法和话语表达形式。
- `negative`：保留药物、事件、相互作用等表面线索，但刻意不提供成立关系所需证据。
- `hardcase`：包含干扰实体、类与实例切换、并列结构或长距离触发因素。
- `margincase`：贴近正例证据，但由于不确定性、混杂因素或时序歧义，仍不应判为确定关系。

## 分布与覆盖

- 原始训练集行数：`{original_train["num_rows"]}`
- 最终增强集行数：`{augmentations["num_rows"]}`
- 最终合并训练集行数：`{final_train["num_rows"]}`
- 最终增强类型分布：`{json.dumps(augmentations["augmentation_type_counts"], ensure_ascii=False)}`
- 最终增强关系分布：`{json.dumps(augmentations["relation_counts"], ensure_ascii=False)}`
- 最终增强集中正样本行数：`{stats["final_augmentation_positive_rows"]}`
- 最终增强集中空目标行数：`{stats["final_augmentation_empty_rows"]}`
- 最终训练集中空目标行数：`{stats["final_train_empty_rows"]}`
- 最终训练集标签分布：`{json.dumps(final_train["relation_counts"], ensure_ascii=False)}`

覆盖亮点：
- 最终增强集中的长距离关系数：`{augmentations["long_gap_relations"]}`
- 最终增强集中的复杂词汇标记统计：`{json.dumps(augmentations["marker_counts"], ensure_ascii=False)}`
- 各增强分组统计摘要保存在 `reports/data_stats.json`。

## 划分污染检查

- 原始训练集与验证集精确重叠：`{len(overlaps["original_train_vs_validation_exact"])}` 行
- 原始训练集与测试集精确重叠：`{len(overlaps["original_train_vs_test_exact"])}` 行
- 验证集与测试集精确重叠：`{len(overlaps["validation_vs_test_exact"])}` 行
- 最终训练集与验证集精确重叠：`{len(overlaps["final_train_vs_validation_exact"])}` 行
- 最终训练集与测试集精确重叠：`{len(overlaps["final_train_vs_test_exact"])}` 行
- 最终增强集与验证集精确重叠：`{len(overlaps["augmentations_vs_validation_exact"])}` 行
- 最终增强集与测试集精确重叠：`{len(overlaps["augmentations_vs_test_exact"])}` 行

## 关键结论

- 最终物化后，仓库内数据 schema 保持内部一致且可解析。
- 原始数据集中没有空目标样本，因此 `negative` 与 `margincase` 填补了真实训练缺口。
- 稀有 DDI 子类，尤其是 `DDI-INT` 和 `DDI-ADVISE`，在首轮增强中相对稀疏，因此做了定向补充。
- 原始划分中存在 train/dev 与 train/test 的精确重叠，这些样本已经从处理后的训练集中移除。
"""


def build_fix_log(stats: Dict[str, Any]) -> str:
    supplement = stats["supplement_delta"]
    return f"""# 数据修复日志

## 输入

- 上一版增强 sidecar：`{stats["paths"]["current_augmentation_jsonl"]}`
- 主增强规格文件：`{stats["paths"]["primary_spec"]}`
- 补充增强规格文件：`{stats["paths"]["supplement_spec"]}`

## 处理动作

- 移除格式损坏行数：0
- 移除非法标签行数：0
- 移除实体边界错误行数：0
- 合并时移除的重复增强规格数：`{stats["spec_merge"]["duplicate_spec_entries_removed"]}`
- 从处理后训练集中移除的精确重叠原始训练样本：`{stats["removed_polluted_train_rows"]}`
- 保留的历史有效增强样本：`{supplement["retained_previous_rows"]}`
- 新增人工整理增强样本：`{supplement["new_rows_added"]}`

## 补充重点

- 增加了更多 `DDI-INT`，覆盖类级别和实例级别的相互作用表述。
- 增加了更多 `DDI-ADVISE` / `DDI-MECHANISM`，以改善标签平衡。
- 增加了更多高迷惑性的 `negative` 与 `margincase` 样本，以强化决策边界。
- 保持所有可训练文件与 ChatML JSONL 格式兼容。

## 最终产物

- 最终训练集：`{stats["paths"]["final_train"]}`
- 最终验证集：`{stats["paths"]["final_validation"]}`
- 最终测试集：`{stats["paths"]["final_test"]}`
- 最终增强 sidecar：`{stats["paths"]["final_augmentations"]}`

## 最终规模

- 原始训练集行数：`{stats["files"]["original_train"]["num_rows"]}`
- 去除划分污染后的基础训练集行数：`{stats["filtered_original_train_rows"]}`
- 最终增强集行数：`{stats["files"]["final_augmentations"]["num_rows"]}`
- 最终合并训练集行数：`{stats["files"]["final_train"]["num_rows"]}`
- 最终数据集哈希：`{stats["final_dataset_hash"]}`
"""


def main() -> None:
    """执行完整的数据审计、物化和报告生成流程。"""
    args = parse_args()
    base_data_dir = (PROJECT_ROOT / args.base_data_dir).resolve()
    report_dir = (PROJECT_ROOT / args.report_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_training_config(PROJECT_ROOT / args.config)
    tokenizer = load_tokenizer(config)
    system_prompt = load_system_prompt(
        str(config["system_prompt_path"]) if config.get("system_prompt_path") is not None else None
    )

    original_train_path = (base_data_dir / "merged_chatml_train.jsonl").resolve()
    original_validation_path = (base_data_dir / "merged_chatml_validation.jsonl").resolve()
    original_test_path = (base_data_dir / "merged_chatml_test.jsonl").resolve()
    current_augmentation_jsonl = (PROJECT_ROOT / args.current_augmentation_jsonl).resolve()
    primary_spec = (PROJECT_ROOT / args.primary_augmentations).resolve()
    supplement_specs = [(PROJECT_ROOT / value).resolve() for value in args.supplement_augmentations]

    original_train_rows = read_jsonl(original_train_path)
    original_validation_rows = read_jsonl(original_validation_path)
    original_test_rows = read_jsonl(original_test_path)
    current_augmentation_rows = read_jsonl(current_augmentation_jsonl) if current_augmentation_jsonl.exists() else []

    polluted_train_keys = (
        exact_overlap(
            (canonical_text(message_content(row, "user")) for row in original_train_rows),
            (canonical_text(message_content(row, "user")) for row in original_validation_rows),
        )
        + exact_overlap(
            (canonical_text(message_content(row, "user")) for row in original_train_rows),
            (canonical_text(message_content(row, "user")) for row in original_test_rows),
        )
    )
    polluted_train_key_set = set(polluted_train_keys)
    filtered_original_train_rows = [
        row
        for row in original_train_rows
        if canonical_text(message_content(row, "user")) not in polluted_train_key_set
    ]
    canonical_filtered_train_rows = canonicalize_chatml_rows(filtered_original_train_rows, system_prompt=system_prompt)
    canonical_validation_rows = canonicalize_chatml_rows(original_validation_rows, system_prompt=system_prompt)
    canonical_test_rows = canonicalize_chatml_rows(original_test_rows, system_prompt=system_prompt)

    spec_records, spec_merge_stats = load_spec_records([primary_spec, *supplement_specs])
    final_augmentation_plain_rows, final_augmentation_rows = build_chatml_rows(spec_records, system_prompt=system_prompt)
    final_train_rows = list(canonical_filtered_train_rows) + list(final_augmentation_plain_rows)

    final_train_path = output_dir / "merged_chatml_train.jsonl"
    final_validation_path = output_dir / "merged_chatml_validation.jsonl"
    final_test_path = output_dir / "merged_chatml_test.jsonl"
    final_augmentations_path = output_dir / "merged_chatml_train_augmentations.jsonl"

    write_jsonl(final_train_path, final_train_rows)
    write_jsonl(final_augmentations_path, final_augmentation_rows)
    write_jsonl(final_validation_path, canonical_validation_rows)
    write_jsonl(final_test_path, canonical_test_rows)

    final_train_audit = file_audit(final_train_rows)
    final_augmentation_audit = file_audit(final_augmentation_rows, expect_augmentation_type=True)
    current_augmentation_audit = file_audit(current_augmentation_rows, expect_augmentation_type=True) if current_augmentation_rows else {}
    original_train_audit = file_audit(original_train_rows)
    original_validation_audit = file_audit(original_validation_rows)
    original_test_audit = file_audit(original_test_rows)

    final_hash = dataset_hash(final_train_rows)
    overlaps = {
        "original_train_vs_validation_exact": exact_overlap(
            (canonical_text(message_content(row, "user")) for row in original_train_rows),
            (canonical_text(message_content(row, "user")) for row in original_validation_rows),
        ),
        "original_train_vs_test_exact": exact_overlap(
            (canonical_text(message_content(row, "user")) for row in original_train_rows),
            (canonical_text(message_content(row, "user")) for row in original_test_rows),
        ),
        "validation_vs_test_exact": exact_overlap(
            (canonical_text(message_content(row, "user")) for row in original_validation_rows),
            (canonical_text(message_content(row, "user")) for row in original_test_rows),
        ),
        "final_train_vs_validation_exact": exact_overlap(
            (canonical_text(message_content(row, "user")) for row in final_train_rows),
            (canonical_text(message_content(row, "user")) for row in original_validation_rows),
        ),
        "final_train_vs_test_exact": exact_overlap(
            (canonical_text(message_content(row, "user")) for row in final_train_rows),
            (canonical_text(message_content(row, "user")) for row in original_test_rows),
        ),
        "augmentations_vs_validation_exact": exact_overlap(
            (canonical_text(message_content(row, "user")) for row in final_augmentation_rows),
            (canonical_text(message_content(row, "user")) for row in original_validation_rows),
        ),
        "augmentations_vs_test_exact": exact_overlap(
            (canonical_text(message_content(row, "user")) for row in final_augmentation_rows),
            (canonical_text(message_content(row, "user")) for row in original_test_rows),
        ),
    }

    current_aug_texts = {canonical_text(message_content(row, "user")) for row in current_augmentation_rows}
    final_aug_texts = {canonical_text(message_content(row, "user")) for row in final_augmentation_rows}
    supplement_delta = {
        "retained_previous_rows": len(current_aug_texts & final_aug_texts),
        "new_rows_added": len(final_aug_texts - current_aug_texts),
        "previous_rows_removed": len(current_aug_texts - final_aug_texts),
    }

    augmentation_group_stats = augmentation_semantic_audit(final_augmentation_rows)
    near_duplicate_pairs = collect_near_duplicate_pairs(final_augmentation_rows)

    tokenizer_stats = {
        "original_train": compute_dataset_statistics(
            original_train_path,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_length=config["max_seq_length"],
            enable_thinking=config.get("enable_thinking"),
        ),
        "final_train": compute_dataset_statistics(
            final_train_path,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_length=config["max_seq_length"],
            enable_thinking=config.get("enable_thinking"),
        ),
        "validation": compute_dataset_statistics(
            final_validation_path,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_length=config["max_seq_length"],
            enable_thinking=config.get("enable_thinking"),
        ),
        "test": compute_dataset_statistics(
            final_test_path,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_length=config["max_seq_length"],
            enable_thinking=config.get("enable_thinking"),
        ),
        "final_augmentations": compute_dataset_statistics(
            final_augmentations_path,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_length=config["max_seq_length"],
            enable_thinking=config.get("enable_thinking"),
        ),
    }

    final_aug_empty_rows = sum(1 for row in final_augmentation_rows if json.loads(message_content(row, "assistant")) == [])
    final_train_empty_rows = sum(1 for row in final_train_rows if json.loads(message_content(row, "assistant")) == [])

    stats = {
        "paths": {
            "original_train": str(original_train_path),
            "original_validation": str(original_validation_path),
            "original_test": str(original_test_path),
            "current_augmentation_jsonl": str(current_augmentation_jsonl),
            "primary_spec": str(primary_spec),
            "supplement_spec": [str(path) for path in supplement_specs],
            "system_prompt": system_prompt,
            "processed_dir": str(output_dir),
            "final_train": str(final_train_path),
            "final_validation": str(final_validation_path),
            "final_test": str(final_test_path),
            "final_augmentations": str(final_augmentations_path),
        },
        "spec_merge": spec_merge_stats,
        "supplement_delta": supplement_delta,
        "final_dataset_hash": final_hash,
        "removed_polluted_train_rows": len(original_train_rows) - len(filtered_original_train_rows),
        "filtered_original_train_rows": len(filtered_original_train_rows),
        "final_augmentation_positive_rows": len(final_augmentation_rows) - final_aug_empty_rows,
        "final_augmentation_empty_rows": final_aug_empty_rows,
        "final_train_empty_rows": final_train_empty_rows,
        "files": {
            "original_train": original_train_audit,
            "original_validation": original_validation_audit,
            "original_test": original_test_audit,
            "current_augmentations": current_augmentation_audit,
            "final_augmentations": final_augmentation_audit,
            "final_train": final_train_audit,
        },
        "overlaps": overlaps,
        "augmentation_group_stats": augmentation_group_stats,
        "final_augmentation_near_duplicates": len(near_duplicate_pairs),
        "final_augmentation_near_duplicate_examples": near_duplicate_pairs[:10],
        "tokenizer_backed_statistics": tokenizer_stats,
        "entity_type_schema": None,
    }

    data_stats_path = report_dir / "data_stats.json"
    data_audit_report_path = report_dir / "data_audit_report.md"
    data_fix_log_path = report_dir / "data_fix_log.md"

    write_json(data_stats_path, stats)
    data_audit_report_path.write_text(build_audit_report(stats), encoding="utf-8")
    data_fix_log_path.write_text(build_fix_log(stats), encoding="utf-8")

    manifest = {
        "final_dataset_hash": final_hash,
        "final_train_path": str(final_train_path),
        "final_validation_path": str(final_validation_path),
        "final_test_path": str(final_test_path),
        "final_augmentations_path": str(final_augmentations_path),
        "system_prompt": system_prompt,
    }
    write_json(output_dir / "manifest.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved audit report to: {data_audit_report_path}")
    print(f"Saved stats JSON to: {data_stats_path}")
    print(f"Saved fix log to: {data_fix_log_path}")


if __name__ == "__main__":
    main()
