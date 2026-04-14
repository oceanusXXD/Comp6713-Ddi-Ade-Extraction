"""Prediction parsing and evaluation utilities.

This module converts raw model output strings into normalized relation records
and computes parse success, exact match, micro precision/recall/F1, and
per-label metrics.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.data_utils import read_jsonl
from src.prompting import extract_message_content

CANONICAL_LABELS: Tuple[str, ...] = (
    "ADE",
    "DDI-MECHANISM",
    "DDI-EFFECT",
    "DDI-ADVISE",
    "DDI-INT",
)

LABEL_ALIASES = {
    "ADE": "ADE",
    "ADVERSE DRUG EVENT": "ADE",
    "DDI-MECHANISM": "DDI-MECHANISM",
    "DDI_MECHANISM": "DDI-MECHANISM",
    "DDI MECHANISM": "DDI-MECHANISM",
    "DDI-MECH": "DDI-MECHANISM",
    "DDI_MECH": "DDI-MECHANISM",
    "MECHANISM": "DDI-MECHANISM",
    "DDI-EFFECT": "DDI-EFFECT",
    "DDI_EFFECT": "DDI-EFFECT",
    "DDI EFFECT": "DDI-EFFECT",
    "EFFECT": "DDI-EFFECT",
    "DDI-ADVISE": "DDI-ADVISE",
    "DDI_ADVISE": "DDI-ADVISE",
    "DDI ADVISE": "DDI-ADVISE",
    "DDI-ADVICE": "DDI-ADVISE",
    "DDI_ADVICE": "DDI-ADVISE",
    "DDI ADVICE": "DDI-ADVISE",
    "ADVISE": "DDI-ADVISE",
    "ADVICE": "DDI-ADVISE",
    "DDI-INT": "DDI-INT",
    "DDI_INT": "DDI-INT",
    "DDI INT": "DDI-INT",
    "INTERACTION": "DDI-INT",
    "INT": "DDI-INT",
}

RELATION_KEYS = {
    "head_entity": ("head_entity", "head", "drug", "entity1", "subject"),
    "tail_entity": ("tail_entity", "tail", "effect", "entity2", "object"),
    "relation_type": ("relation_type", "label", "relation", "type"),
}

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)


@dataclass
class ParsedPrediction:
    """Normalized parsed representation of one model output."""
    relations: List[Dict[str, str]]
    status: str
    failure_reason: Optional[str]
    raw_candidate: Optional[str]


@dataclass
class DatasetExample:
    """Lightweight sample object used during inference."""
    sample_id: str
    split: str
    system_prompt: str
    user_text: str
    gold_relations: List[Dict[str, str]]


def normalize_text(value: Any) -> str:
    """Collapse whitespace to reduce alignment noise from formatting differences."""
    return " ".join(str(value).strip().split())


def normalize_label(raw_label: Any) -> str:
    """Map label variants from model output to canonical labels."""
    normalized = normalize_text(raw_label).replace("_", " ").replace("-", " ").upper()
    normalized = " ".join(normalized.split())
    if normalized not in LABEL_ALIASES:
        raise ValueError(f"Unsupported relation label: {raw_label!r}")
    return LABEL_ALIASES[normalized]


def normalize_relation_item(item: Dict[str, Any]) -> Dict[str, str]:
    """Normalize relation field aliases to the repository field names."""
    if not isinstance(item, dict):
        raise ValueError(f"Unsupported relation item type: {type(item)!r}")
    normalized_item: Dict[str, str] = {}
    for target_key, aliases in RELATION_KEYS.items():
        raw_value: Any = ""
        for alias in aliases:
            if alias in item:
                raw_value = item[alias]
                break
        if target_key == "relation_type":
            normalized_item[target_key] = normalize_label(raw_value)
        else:
            normalized_item[target_key] = normalize_text(raw_value)
    return normalized_item


def normalize_relation_list(relations: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize, deduplicate, and stably sort relation lists."""
    deduplicated: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for relation in relations:
        normalized = normalize_relation_item(relation)
        key = (
            normalized["relation_type"],
            normalized["head_entity"].casefold(),
            normalized["tail_entity"].casefold(),
        )
        deduplicated[key] = normalized
    return [deduplicated[key] for key in sorted(deduplicated)]


def serialize_relations(relations: Sequence[Dict[str, Any]]) -> str:
    """Serialize relation lists into stable JSON for comparison and caching."""
    return json.dumps(
        normalize_relation_list(relations),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _strip_generation_noise(text: str) -> str:
    """Remove `<think>` blocks, markdown fences, and similar generation noise."""
    cleaned = THINK_BLOCK_RE.sub("", text or "").strip()
    if not cleaned:
        return cleaned
    fence_match = CODE_FENCE_RE.search(cleaned)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def _iter_json_candidates(text: str) -> Iterable[str]:
    """Enumerate likely JSON candidate fragments from raw output."""
    cleaned = _strip_generation_noise(text)
    if not cleaned:
        return

    yield cleaned

    decoder = json.JSONDecoder()
    for opening_char in ("[", "{"):
        start_index = 0
        while True:
            start_index = cleaned.find(opening_char, start_index)
            if start_index == -1:
                break
            try:
                _, end_index = decoder.raw_decode(cleaned, start_index)
            except json.JSONDecodeError:
                start_index += 1
                continue
            candidate = cleaned[start_index:end_index].strip()
            if candidate and candidate != cleaned:
                yield candidate
            start_index += 1


def _coerce_candidate_to_relation_payload(candidate: Any) -> List[Dict[str, Any]]:
    """Extract a relation list from different candidate JSON shapes."""
    if isinstance(candidate, list):
        return candidate
    if isinstance(candidate, dict):
        for key in ("relations", "outputs", "predictions", "items", "data"):
            nested = candidate.get(key)
            if isinstance(nested, list):
                return nested
        required = {"head_entity", "tail_entity", "relation_type"}
        if required.issubset(candidate):
            return [candidate]
    raise ValueError(f"Unsupported prediction payload type: {type(candidate)!r}")


def parse_prediction_text(text: str) -> ParsedPrediction:
    """Recover a relation list from model output on a best-effort basis.

    The parser removes generation wrappers, enumerates JSON candidates, tries
    `json.loads` first, falls back to `ast.literal_eval`, and then normalizes
    fields and labels.
    """
    cleaned = _strip_generation_noise(text)
    if not cleaned:
        return ParsedPrediction(relations=[], status="parse_failure", failure_reason="empty_output", raw_candidate=None)

    last_error: Optional[Exception] = None
    raw_candidate: Optional[str] = None

    for candidate_text in _iter_json_candidates(cleaned):
        raw_candidate = candidate_text
        for loader in (json.loads, ast.literal_eval):
            try:
                candidate = loader(candidate_text)
                relations = normalize_relation_list(_coerce_candidate_to_relation_payload(candidate))
                return ParsedPrediction(
                    relations=relations,
                    status="parsed",
                    failure_reason=None,
                    raw_candidate=candidate_text,
                )
            except (ValueError, TypeError, SyntaxError, json.JSONDecodeError) as exc:
                last_error = exc
                continue

    failure_reason = "invalid_json"
    if isinstance(last_error, ValueError) and "Unsupported relation label" in str(last_error):
        failure_reason = "invalid_label"
    elif isinstance(last_error, ValueError):
        failure_reason = "invalid_schema"

    return ParsedPrediction(
        relations=[],
        status="parse_failure",
        failure_reason=failure_reason,
        raw_candidate=raw_candidate,
    )


def load_dataset_examples(path: Path, *, split: str, limit: Optional[int] = None) -> List[DatasetExample]:
    """Load ChatML JSONL rows as `DatasetExample` objects for inference or evaluation."""
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]

    examples: List[DatasetExample] = []
    for index, row in enumerate(rows):
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError(f"Row {index} in {path} is missing a valid messages list.")

        system_prompt = extract_message_content(messages, "system")
        user_text = extract_message_content(messages, "user")
        assistant_content = extract_message_content(messages, "assistant")
        if not system_prompt or not user_text:
            raise ValueError(f"Row {index} in {path} is missing system/user content.")

        gold_relations = []
        if assistant_content != "":
            gold_relations = parse_prediction_text(assistant_content).relations

        examples.append(
            DatasetExample(
                sample_id=f"{split}_{index:04d}",
                split=split,
                system_prompt=system_prompt,
                user_text=user_text,
                gold_relations=gold_relations,
            )
        )
    return examples


def relation_set(relations: Sequence[Dict[str, Any]]) -> set[Tuple[str, str, str]]:
    """Convert relation lists to sets for set-level comparison."""
    normalized = normalize_relation_list(relations)
    return {
        (
            relation["relation_type"],
            relation["head_entity"].casefold(),
            relation["tail_entity"].casefold(),
        )
        for relation in normalized
    }


def evaluate_prediction_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute parse rate, exact match, and micro/per-label metrics over rows."""
    total_samples = len(rows)
    parsed_samples = 0
    exact_matches = 0
    tp = 0
    fp = 0
    fn = 0
    failure_counts: Counter[str] = Counter()
    gold_label_counts: Counter[str] = Counter()
    pred_label_counts: Counter[str] = Counter()
    per_label_tp: Counter[str] = Counter()
    per_label_fp: Counter[str] = Counter()
    per_label_fn: Counter[str] = Counter()

    for row in rows:
        gold_relations = normalize_relation_list(row.get("gold_relations", []))
        predicted_relations = normalize_relation_list(row.get("predicted_relations", []))
        parse_status = str(row.get("parse_status", "parsed"))
        failure_reason = row.get("parse_failure_reason")

        if parse_status == "parsed":
            parsed_samples += 1
        elif failure_reason:
            failure_counts[str(failure_reason)] += 1
        else:
            failure_counts["unknown"] += 1

        gold_set = relation_set(gold_relations)
        pred_set = relation_set(predicted_relations)

        if gold_set == pred_set:
            exact_matches += 1

        # Use set-level comparison and require an exact label/head/tail match.
        row_tp = gold_set & pred_set
        row_fp = pred_set - gold_set
        row_fn = gold_set - pred_set
        tp += len(row_tp)
        fp += len(row_fp)
        fn += len(row_fn)

        for relation in gold_relations:
            gold_label_counts[relation["relation_type"]] += 1
        for relation in predicted_relations:
            pred_label_counts[relation["relation_type"]] += 1
        for label, _, _ in row_tp:
            per_label_tp[label] += 1
        for label, _, _ in row_fp:
            per_label_fp[label] += 1
        for label, _, _ in row_fn:
            per_label_fn[label] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    per_label_metrics: Dict[str, Dict[str, float]] = {}
    for label in CANONICAL_LABELS:
        label_tp = per_label_tp[label]
        label_fp = per_label_fp[label]
        label_fn = per_label_fn[label]
        label_precision = label_tp / (label_tp + label_fp) if (label_tp + label_fp) else 0.0
        label_recall = label_tp / (label_tp + label_fn) if (label_tp + label_fn) else 0.0
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if (label_precision + label_recall)
            else 0.0
        )
        per_label_metrics[label] = {
            "tp": label_tp,
            "fp": label_fp,
            "fn": label_fn,
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
            "gold_count": gold_label_counts[label],
            "pred_count": pred_label_counts[label],
        }

    return {
        "total_samples": total_samples,
        "parsed_samples": parsed_samples,
        "parse_success_rate": (parsed_samples / total_samples) if total_samples else 0.0,
        "parse_failure_counts": dict(sorted(failure_counts.items())),
        "exact_match_count": exact_matches,
        "exact_match_accuracy": (exact_matches / total_samples) if total_samples else 0.0,
        "micro": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "per_label": per_label_metrics,
    }


def canonicalize_prediction_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize historical prediction formats into the current evaluation format."""
    normalized_row = dict(row)

    gold_relations = row.get("gold_relations", [])
    if not isinstance(gold_relations, list):
        gold_relations = []

    if "predicted_relations" in row and isinstance(row["predicted_relations"], list):
        predicted_relations = row["predicted_relations"]
        parse_status = str(row.get("parse_status", "parsed"))
        parse_failure_reason = row.get("parse_failure_reason")
    elif "parsed_output" in row:
        parsed_output = row.get("parsed_output")
        predicted_relations = parsed_output if isinstance(parsed_output, list) else []
        json_valid = bool(row.get("json_valid"))
        parse_status = "parsed" if json_valid else "parse_failure"
        parse_failure_reason = None if json_valid else "legacy_invalid_json"
    else:
        parsed_prediction = parse_prediction_text(str(row.get("raw_output", "")))
        predicted_relations = parsed_prediction.relations
        parse_status = parsed_prediction.status
        parse_failure_reason = parsed_prediction.failure_reason
        normalized_row.setdefault("raw_json_candidate", parsed_prediction.raw_candidate)

    normalized_row["gold_relations"] = normalize_relation_list(gold_relations)
    normalized_row["predicted_relations"] = normalize_relation_list(predicted_relations)
    normalized_row["parse_status"] = parse_status
    normalized_row["parse_failure_reason"] = parse_failure_reason
    return normalized_row


def format_metrics_report(metrics: Dict[str, Any], *, prediction_path: Optional[Path] = None) -> str:
    """Format a metrics dictionary as a human-readable text report."""
    lines = ["=" * 80]
    if prediction_path is not None:
        lines.append(f"Prediction file: {prediction_path}")
    lines.append(f"Total samples: {metrics['total_samples']}")
    lines.append(
        "Parsed successfully: "
        f"{metrics['parsed_samples']}/{metrics['total_samples']} = {metrics['parse_success_rate']:.2%}"
    )
    if metrics["parse_failure_counts"]:
        lines.append(f"Parse failure counts: {metrics['parse_failure_counts']}")
    lines.append(
        "Exact match accuracy: "
        f"{metrics['exact_match_count']}/{metrics['total_samples']} = {metrics['exact_match_accuracy']:.2%}"
    )
    micro = metrics["micro"]
    lines.append(f"TP={micro['tp']}, FP={micro['fp']}, FN={micro['fn']}")
    lines.append(f"Precision: {micro['precision']:.4f}")
    lines.append(f"Recall:    {micro['recall']:.4f}")
    lines.append(f"F1:        {micro['f1']:.4f}")
    lines.append("-" * 80)
    lines.append("Per-label metrics:")
    for label in CANONICAL_LABELS:
        label_metrics = metrics["per_label"][label]
        lines.append(
            f"{label}: TP={label_metrics['tp']} FP={label_metrics['fp']} FN={label_metrics['fn']} "
            f"P={label_metrics['precision']:.4f} R={label_metrics['recall']:.4f} F1={label_metrics['f1']:.4f}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)
