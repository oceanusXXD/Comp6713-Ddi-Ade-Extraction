"""数据读取、标签规范化和监督训练样本构造工具。

这个模块承担三类职责：
1. 读取 ChatML JSONL 并把 assistant 输出标准化为统一关系格式。
2. 统计数据集分布、长度和标签信息，供训练前审计与观测使用。
3. 把一条 ChatML 样本编码成监督微调可直接消费的 token / label 张量。
"""

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
    """逐行读取 JSONL 文件并返回字典列表。"""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(value: Any) -> str:
    """做最轻量的文本规范化，主要用于去除多余空白。"""
    return " ".join(str(value).strip().split())


def normalize_label_key(raw_label: Any) -> str:
    """把标签统一成便于查表的 key 形式。"""
    label = normalize_text(raw_label)
    return re.sub(r"[-_\s]+", "-", label).upper()


def normalize_label(raw_label: Any) -> str:
    """把各种历史标签别名映射到仓库内部规范标签。"""
    canonical_key = normalize_label_key(raw_label)
    if canonical_key not in LABEL_NORMALIZATION:
        raise ValueError(f"Unsupported relation label: {raw_label!r}")
    return LABEL_NORMALIZATION[canonical_key]


def parse_assistant_relations(assistant_content: str) -> List[Dict[str, str]]:
    """解析 assistant 的 JSON 字符串输出，并做字段与标签规范化。"""
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
    """对关系列表做去重、清洗和稳定排序。"""
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
    """把规范化关系列表序列化成稳定 JSON 字符串。"""
    normalized = normalize_relation_list(relations)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_training_example(row: Dict[str, Any], system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """从一条 ChatML 记录抽取出训练时真正需要的三段文本和 gold 关系。"""
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
    """统计数据集中样本数、空目标数和标签分布。"""
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
    """在纯 Python 环境下计算稳健分位数，避免引入额外数值依赖。"""
    if not values:
        return 0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * percentile))))
    return sorted_values[index]


def _summarize_numeric(values: Sequence[int]) -> Dict[str, float]:
    """把一组数值压缩成常见统计摘要，便于写入报告。"""
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
    """计算数据集长度、标签和 token 侧统计信息。

    如果传入 tokenizer，这个函数还会额外统计 prompt 长度、完整样本长度，
    以及样本是否超过训练时 `max_length`。
    """
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
            # 这里分别统计“只有 prompt”与“prompt + assistant 目标”两种长度，
            # 这样后面可以更容易定位是输入过长，还是目标答案本身过长。
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
    """把单条监督样本编码成训练输入。

    返回 `None` 代表该样本无法安全用于当前训练设置，最常见原因是：
    - 编码后长度超过 `max_length`
    - assistant 区域无法可靠定位
    - 生成出的 label 全部被 mask 掉
    """
    full_messages = build_messages(system_prompt, user_text, target_text)
    prompt_messages = build_messages(system_prompt, user_text)

    if supports_assistant_tokens_mask(tokenizer):
        try:
            # 如果 tokenizer 原生支持 assistant mask，就优先走这条路径。
            # 这样能更精确地只监督 assistant 输出部分，避免手动估计边界。
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
            # 某些 tokenizer 虽然暴露了相关接口，但运行时并不稳定；
            # 这里回退到“prompt 前缀 + assistant 尾部”差分策略。
            pass

    # 回退路径：先分别编码 prompt 与 full messages，再用长度差构造 labels。
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
    """记录数据集构造过程中最关心的几个计数指标。"""
    num_rows: int = 0
    num_encoded: int = 0
    num_skipped_over_max_length: int = 0


class SupervisedChatDataset(Dataset):
    """简单包装编码后样本，供 Hugging Face Trainer 直接读取。"""
    def __init__(self, items: List[Dict[str, List[int]]], sample_weights: Optional[List[float]] = None):
        self.items = items
        self.sample_weights = sample_weights if sample_weights is not None else [1.0] * len(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.items[index]


class SupervisedDataCollator:
    """把不同长度的监督样本 pad 成 batch。

    除了标准的 `input_ids / attention_mask / labels`，这里还会保留
    每条样本的 `loss_weights`，供自定义 loss 计算时使用。
    """
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define a pad_token_id for batching.")

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """把一个 batch 的变长样本对齐到同一长度。"""
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
            "loss_weights": torch.tensor(
                [float(feature.get("loss_weight", 1.0)) for feature in features],
                dtype=torch.float,
            ),
        }


def compute_relation_weight(
    relations: Sequence[Dict[str, str]],
    *,
    empty_target_weight: float = 1.0,
    ddi_weight: float = 1.0,
    ddi_int_weight: float = 1.0,
    multi_relation_weight: float = 1.0,
) -> float:
    """根据关系类型和样本难度启发式计算权重。

    这里同一套逻辑同时被用于：
    - 采样权重：决定训练时更容易抽到哪些样本
    - loss 权重：决定不同样本对总 loss 的贡献大小
    """
    weight = 1.0
    if not relations:
        weight *= float(empty_target_weight)
    if any(relation["relation_type"] != "ADE" for relation in relations):
        weight *= float(ddi_weight)
    if any(relation["relation_type"] == "DDI-INT" for relation in relations):
        weight *= float(ddi_int_weight)
    if len(relations) > 1:
        weight *= float(multi_relation_weight)
    return weight


def build_supervised_dataset(
    path: Path,
    tokenizer: Any,
    *,
    system_prompt: str,
    max_length: int,
    enable_thinking: Optional[bool],
    limit: Optional[int] = None,
    empty_target_sampling_weight: float = 1.0,
    ddi_sampling_weight: float = 1.0,
    ddi_int_sampling_weight: float = 1.0,
    multi_relation_sampling_weight: float = 1.0,
    empty_target_loss_weight: float = 1.0,
    ddi_loss_weight: float = 1.0,
    ddi_int_loss_weight: float = 1.0,
    multi_relation_loss_weight: float = 1.0,
) -> Tuple[SupervisedChatDataset, DatasetBuildStats]:
    """从 ChatML JSONL 构建监督训练数据集。

    这个函数会串起：
    - 读取原始样本
    - 提取训练字段
    - token 化
    - 长度过滤
    - 采样 / loss 权重计算
    """
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]

    stats = DatasetBuildStats(num_rows=len(rows))
    items: List[Dict[str, List[int]]] = []
    sample_weights: List[float] = []
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
        relations = example["relations"]
        sample_weight = compute_relation_weight(
            relations,
            empty_target_weight=empty_target_sampling_weight,
            ddi_weight=ddi_sampling_weight,
            ddi_int_weight=ddi_int_sampling_weight,
            multi_relation_weight=multi_relation_sampling_weight,
        )
        tokenized["loss_weight"] = compute_relation_weight(
            relations,
            empty_target_weight=empty_target_loss_weight,
            ddi_weight=ddi_loss_weight,
            ddi_int_weight=ddi_int_loss_weight,
            multi_relation_weight=multi_relation_loss_weight,
        )
        items.append(tokenized)
        sample_weights.append(sample_weight)
        stats.num_encoded += 1

    return SupervisedChatDataset(items, sample_weights=sample_weights), stats
