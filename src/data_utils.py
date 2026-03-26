import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from src.prompting import apply_chat_template, build_messages, extract_message_content, supports_assistant_tokens_mask

CANONICAL_LABELS = (
    "ADE",
    "DDI-MECHANISM",
    "DDI-EFFECT",
    "DDI-ADVISE",
    "DDI-INT",
)

LABEL_NORMALIZATION = {
    "ADE": "ADE",
    "ADVERSE-DRUG-EVENT": "ADE",
    "DDI-MECHANISM": "DDI-MECHANISM",
    "DDI-MECH": "DDI-MECHANISM",
    "DDI-EFFECT": "DDI-EFFECT",
    "DDI-ADVISE": "DDI-ADVISE",
    "DDI-ADVICE": "DDI-ADVISE",
    "DDI-INT": "DDI-INT",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def normalize_label_key(raw_label: Any) -> str:
    label = normalize_text(raw_label)
    return re.sub(r"[-_\s]+", "-", label).upper()


def normalize_label(raw_label: Any) -> str:
    canonical_key = normalize_label_key(raw_label)
    if canonical_key not in LABEL_NORMALIZATION:
        raise ValueError(f"Unsupported relation label: {raw_label!r}")
    return LABEL_NORMALIZATION[canonical_key]


def parse_assistant_relations(assistant_content: str) -> List[Dict[str, str]]:
    parsed = json.loads(assistant_content)
    if not isinstance(parsed, list):
        raise ValueError("Assistant content must be a JSON list.")
    normalized_records: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(f"Assistant relation must be an object, got: {type(item)!r}")
        normalized_records.append(
            {
                "head_entity": normalize_text(item.get("head_entity", "")),
                "tail_entity": normalize_text(item.get("tail_entity", "")),
                "relation_type": normalize_label(item.get("relation_type", "")),
            }
        )
    return normalize_relation_list(normalized_records)


def normalize_relation_list(relations: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    deduplicated: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for relation in relations:
        head = normalize_text(relation.get("head_entity", ""))
        tail = normalize_text(relation.get("tail_entity", ""))
        label = normalize_label(relation.get("relation_type", ""))
        key = (label, head.casefold(), tail.casefold())
        deduplicated[key] = {
            "head_entity": head,
            "tail_entity": tail,
            "relation_type": label,
        }
    return [deduplicated[key] for key in sorted(deduplicated)]


def serialize_relations(relations: Sequence[Dict[str, str]]) -> str:
    normalized = normalize_relation_list(relations)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_training_example(row: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("Each row must contain a list under 'messages'.")

    resolved_system_prompt = system_prompt if system_prompt is not None else extract_message_content(messages, "system")
    user_text = extract_message_content(messages, "user")
    assistant_content = extract_message_content(messages, "assistant")
    if not resolved_system_prompt or not user_text or assistant_content == "":
        raise ValueError("Each training row must contain system, user, and assistant messages.")

    relations = parse_assistant_relations(assistant_content)
    target_text = serialize_relations(relations)
    return {
        "system_prompt": resolved_system_prompt,
        "user_text": user_text,
        "target_text": target_text,
        "relations": relations,
    }


def summarize_chat_dataset(path: Path) -> Dict[str, Any]:
    rows = read_jsonl(path)
    label_counts: Counter[str] = Counter()
    empty_targets = 0
    for row in rows:
        example = extract_training_example(row)
        if not example["relations"]:
            empty_targets += 1
            continue
        for relation in example["relations"]:
            label_counts[relation["relation_type"]] += 1

    return {
        "path": str(path),
        "num_rows": len(rows),
        "num_empty_targets": empty_targets,
        "label_counts": dict(sorted(label_counts.items())),
    }


def _safe_percentile(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * percentile))))
    return sorted_values[index]


def _summarize_numeric(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
        }

    sorted_values = sorted(values)
    total = sum(sorted_values)
    count = len(sorted_values)
    mid = count // 2
    if count % 2 == 1:
        median = float(sorted_values[mid])
    else:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2.0

    return {
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": total / count,
        "median": median,
        "p90": _safe_percentile(sorted_values, 0.90),
        "p95": _safe_percentile(sorted_values, 0.95),
        "p99": _safe_percentile(sorted_values, 0.99),
    }


def compute_dataset_statistics(
    path: Path,
    *,
    tokenizer: Any = None,
    system_prompt: Optional[str] = None,
    max_length: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]

    label_counts: Counter[str] = Counter()
    relations_per_example: List[int] = []
    user_word_lengths: List[int] = []
    user_char_lengths: List[int] = []
    target_char_lengths: List[int] = []
    prompt_token_lengths: List[int] = []
    full_token_lengths: List[int] = []
    target_token_lengths: List[int] = []
    examples_over_max_length = 0
    duplicate_relation_examples = 0
    raw_relation_total = 0
    normalized_relation_total = 0

    for row in rows:
        messages = row.get("messages", [])
        raw_assistant_content = extract_message_content(messages, "assistant")
        raw_relations = json.loads(raw_assistant_content)
        if len(raw_relations) != len(normalize_relation_list(raw_relations)):
            duplicate_relation_examples += 1
        raw_relation_total += len(raw_relations)

        example = extract_training_example(row, system_prompt=system_prompt)
        relations = example["relations"]
        target_text = example["target_text"]
        user_text = example["user_text"]

        normalized_relation_total += len(relations)
        relations_per_example.append(len(relations))
        user_word_lengths.append(len(user_text.split()))
        user_char_lengths.append(len(user_text))
        target_char_lengths.append(len(target_text))

        for relation in relations:
            label_counts[relation["relation_type"]] += 1

        if tokenizer is not None:
            prompt_messages = build_messages(example["system_prompt"], user_text)
            full_messages = build_messages(example["system_prompt"], user_text, target_text)
            prompt_ids = apply_chat_template(
                tokenizer,
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=False,
                enable_thinking=enable_thinking,
            )
            full_ids = apply_chat_template(
                tokenizer,
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=False,
                enable_thinking=enable_thinking,
            )
            prompt_len = len(prompt_ids)
            full_len = len(full_ids)
            prompt_token_lengths.append(prompt_len)
            full_token_lengths.append(full_len)
            target_token_lengths.append(max(0, full_len - min(prompt_len, full_len)))
            if max_length is not None and full_len > max_length:
                examples_over_max_length += 1

    stats: Dict[str, Any] = {
        "path": str(path),
        "num_rows": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "raw_relation_total": raw_relation_total,
        "normalized_relation_total": normalized_relation_total,
        "duplicate_relation_examples": duplicate_relation_examples,
        "relations_per_example": _summarize_numeric(relations_per_example),
        "user_word_lengths": _summarize_numeric(user_word_lengths),
        "user_char_lengths": _summarize_numeric(user_char_lengths),
        "target_char_lengths": _summarize_numeric(target_char_lengths),
    }

    if tokenizer is not None:
        stats["prompt_token_lengths"] = _summarize_numeric(prompt_token_lengths)
        stats["full_token_lengths"] = _summarize_numeric(full_token_lengths)
        stats["target_token_lengths"] = _summarize_numeric(target_token_lengths)
        stats["examples_over_max_length"] = examples_over_max_length
        stats["examples_over_max_length_rate"] = (
            examples_over_max_length / len(rows) if rows else 0.0
        )
        stats["max_length"] = max_length

    return stats


def tokenize_supervised_example(
    tokenizer: Any,
    system_prompt: str,
    user_text: str,
    target_text: str,
    *,
    max_length: int,
    enable_thinking: Optional[bool],
) -> Optional[Dict[str, List[int]]]:
    full_messages = build_messages(system_prompt, user_text, target_text)
    prompt_messages = build_messages(system_prompt, user_text)

    if supports_assistant_tokens_mask(tokenizer):
        try:
            encoded = apply_chat_template(
                tokenizer,
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
                enable_thinking=enable_thinking,
            )
            input_ids = list(encoded["input_ids"])
            if len(input_ids) > max_length:
                return None
            attention_mask = list(encoded["attention_mask"])
            assistant_mask = list(encoded.get("assistant_masks", []))
            if assistant_mask:
                labels = [
                    token_id if assistant_flag else -100
                    for token_id, assistant_flag in zip(input_ids, assistant_mask)
                ]
                if any(label != -100 for label in labels):
                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    }
        except Exception:
            pass

    prompt_ids = apply_chat_template(
        tokenizer,
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        truncation=False,
        enable_thinking=enable_thinking,
    )
    full_ids = apply_chat_template(
        tokenizer,
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=False,
        enable_thinking=enable_thinking,
    )
    if len(full_ids) > max_length:
        return None

    input_ids = list(full_ids)
    prompt_length = min(len(prompt_ids), len(input_ids))
    if prompt_length >= len(input_ids):
        return None
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


@dataclass
class DatasetBuildStats:
    num_rows: int = 0
    num_encoded: int = 0
    num_skipped_over_max_length: int = 0


class SupervisedChatDataset(Dataset):
    def __init__(self, items: List[Dict[str, List[int]]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.items[index]


class SupervisedDataCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define a pad_token_id for batching.")

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            pad_size = max_length - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_size)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_size)
            batch_labels.append(feature["labels"] + [-100] * pad_size)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def build_supervised_dataset(
    path: Path,
    tokenizer: Any,
    *,
    system_prompt: str,
    max_length: int,
    enable_thinking: Optional[bool],
    limit: Optional[int] = None,
) -> Tuple[SupervisedChatDataset, DatasetBuildStats]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]

    stats = DatasetBuildStats(num_rows=len(rows))
    items: List[Dict[str, List[int]]] = []
    for row in rows:
        example = extract_training_example(row, system_prompt=system_prompt)
        tokenized = tokenize_supervised_example(
            tokenizer,
            example["system_prompt"],
            example["user_text"],
            example["target_text"],
            max_length=max_length,
            enable_thinking=enable_thinking,
        )
        if tokenized is None:
            stats.num_skipped_over_max_length += 1
            continue
        items.append(tokenized)
        stats.num_encoded += 1

    return SupervisedChatDataset(items), stats
