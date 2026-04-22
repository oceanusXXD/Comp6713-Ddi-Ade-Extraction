"""Prediction parsing, dataset loading, and evaluation helpers."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.data_utils import (
    CANONICAL_LABELS,
    normalize_relation_list,
    normalize_text,
    parse_assistant_relations,
    read_jsonl,
)
from src.prompting import extract_message_content

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class DatasetExample:
    """A normalized inference-time example."""

    sample_id: str
    split: str
    system_prompt: str
    user_text: str
    gold_relations: Optional[List[Dict[str, str]]]


@dataclass(frozen=True)
class ParsedPrediction:
    """The parsed form of one model output string."""

    relations: List[Dict[str, str]]
    status: str
    failure_reason: Optional[str]
    raw_candidate: Optional[str]


def serialize_relations(relations: Sequence[Dict[str, Any]]) -> str:
    """Serialize normalized relations into stable JSON."""

    return json.dumps(
        normalize_relation_list(relations),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _strip_generation_noise(text: str) -> str:
    """Remove common non-JSON wrappers from generation output."""

    cleaned = THINK_BLOCK_RE.sub("", text or "").strip()
    if not cleaned:
        return ""

    fence_match = CODE_FENCE_RE.search(cleaned)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def _iter_candidate_texts(text: str) -> Iterable[str]:
    """Yield possible JSON payload substrings from a raw generation string."""

    seen: set[str] = set()

    def add(value: str) -> Iterable[str]:
        candidate = value.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate

    yield from add(text)

    for opening, closing in (("[", "]"), ("{", "}")):
        start = text.find(opening)
        end = text.rfind(closing)
        if start != -1 and end > start:
            yield from add(text[start : end + 1])

        depth = 0
        first_start: Optional[int] = None
        for index, char in enumerate(text):
            if char == opening:
                if depth == 0:
                    first_start = index
                depth += 1
            elif char == closing and depth > 0:
                depth -= 1
                if depth == 0 and first_start is not None:
                    yield from add(text[first_start : index + 1])
                    break


def _coerce_candidate_to_relation_payload(candidate: Any) -> List[Dict[str, Any]]:
    """Coerce a parsed JSON payload into a list of relation-like dicts."""

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
    """Parse one model output string into normalized relation triples."""

    cleaned = _strip_generation_noise(text)
    if not cleaned:
        return ParsedPrediction(
            relations=[],
            status="parse_failure",
            failure_reason="empty_output",
            raw_candidate=None,
        )

    last_error: Optional[Exception] = None
    raw_candidate: Optional[str] = None

    for candidate_text in _iter_candidate_texts(cleaned):
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


def load_dataset_examples(path: Path | str, *, split: str, limit: Optional[int] = None) -> List[DatasetExample]:
    """Load packaged ChatML examples for inference or evaluation."""

    path = Path(path)
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]

    examples: List[DatasetExample] = []
    normalized_split = str(split).strip() or "dataset"

    for index, row in enumerate(rows):
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("Each dataset row must contain a list under 'messages'.")

        system_prompt = extract_message_content(messages, "system")
        user_text = extract_message_content(messages, "user")
        assistant_content = extract_message_content(messages, "assistant")
        if not system_prompt or not user_text:
            raise ValueError("Each dataset row must contain system and user messages.")

        gold_relations = parse_assistant_relations(assistant_content) if assistant_content else []
        sample_id = str(
            row.get("sample_id")
            or row.get("id")
            or row.get("uid")
            or f"{normalized_split}_{index:04d}"
        )
        examples.append(
            DatasetExample(
                sample_id=sample_id,
                split=normalized_split,
                system_prompt=system_prompt,
                user_text=user_text,
                gold_relations=gold_relations,
            )
        )

    return examples


def canonicalize_prediction_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one prediction row into the repository's canonical schema."""

    normalized_row = dict(row)
    normalized_row["sample_id"] = str(normalized_row.get("sample_id") or "")
    normalized_row["split"] = str(normalized_row.get("split") or "")
    normalized_row["text"] = normalize_text(normalized_row.get("text", ""))
    normalized_row["system_prompt"] = str(normalized_row.get("system_prompt") or "").strip()
    normalized_row["raw_output"] = str(normalized_row.get("raw_output") or "")
    normalized_row["raw_json_candidate"] = normalized_row.get("raw_json_candidate")
    normalized_row["model_name_or_path"] = (
        None if normalized_row.get("model_name_or_path") is None else str(normalized_row["model_name_or_path"])
    )
    normalized_row["adapter_path"] = (
        None if normalized_row.get("adapter_path") is None else str(normalized_row["adapter_path"])
    )

    gold_relations = normalized_row.get("gold_relations")
    if gold_relations is None:
        normalized_row["gold_relations"] = None
    else:
        normalized_row["gold_relations"] = normalize_relation_list(gold_relations)

    parse_status = normalized_row.get("parse_status")
    parse_failure_reason = normalized_row.get("parse_failure_reason")
    predicted_relations = normalized_row.get("predicted_relations")

    if not isinstance(predicted_relations, list) and "parsed_output" in normalized_row:
        legacy_parsed_output = normalized_row.get("parsed_output")
        if isinstance(legacy_parsed_output, list):
            predicted_relations = legacy_parsed_output
        else:
            predicted_relations = []
        parse_status = "parsed" if bool(normalized_row.get("json_valid")) else "parse_failure"
        parse_failure_reason = None if parse_status == "parsed" else "legacy_invalid_json"

    reparsed: Optional[ParsedPrediction] = None
    if isinstance(predicted_relations, list):
        try:
            normalized_row["predicted_relations"] = normalize_relation_list(predicted_relations)
        except (TypeError, ValueError):
            reparsed = parse_prediction_text(normalized_row["raw_output"])
    else:
        reparsed = parse_prediction_text(normalized_row["raw_output"])

    if reparsed is not None:
        normalized_row["predicted_relations"] = reparsed.relations
        normalized_row["raw_json_candidate"] = reparsed.raw_candidate
        parse_status = reparsed.status
        parse_failure_reason = reparsed.failure_reason
    else:
        normalized_row.setdefault("predicted_relations", [])
        if parse_status is None:
            parse_status = "parsed"
        if parse_status == "parsed":
            parse_failure_reason = None

    normalized_row["parse_status"] = str(parse_status)
    normalized_row["parse_failure_reason"] = (
        None if parse_failure_reason is None else str(parse_failure_reason)
    )
    return normalized_row


def _relation_set(relations: Sequence[Dict[str, Any]]) -> set[Tuple[str, str, str]]:
    """Convert relations into a comparable triple set."""

    relation_set: set[Tuple[str, str, str]] = set()
    for relation in normalize_relation_list(relations):
        relation_set.add(
            (
                relation["relation_type"],
                relation["head_entity"].casefold(),
                relation["tail_entity"].casefold(),
            )
        )
    return relation_set


def relation_set(relations: Sequence[Dict[str, Any]]) -> set[Tuple[str, str, str]]:
    """Public wrapper for relation-set comparison used by the Gradio demo."""
    return _relation_set(relations)


def evaluate_prediction_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate parse and extraction metrics over prediction rows."""

    normalized_rows = [canonicalize_prediction_row(row) for row in rows]
    total_samples = len(normalized_rows)
    parsed_samples = sum(1 for row in normalized_rows if row["parse_status"] == "parsed")
    nonempty_samples = sum(1 for row in normalized_rows if row["predicted_relations"])
    predicted_relation_total = sum(len(row["predicted_relations"]) for row in normalized_rows)

    failure_counts: Counter[str] = Counter()
    evaluation_samples = 0
    exact_match_count = 0
    tp = 0
    fp = 0
    fn = 0

    for row in normalized_rows:
        if row["parse_status"] != "parsed":
            failure_counts[row["parse_failure_reason"] or "unknown"] += 1

        gold_relations = row.get("gold_relations")
        if gold_relations is None:
            continue

        evaluation_samples += 1
        gold_set = _relation_set(gold_relations)
        predicted_set = _relation_set(row["predicted_relations"])
        if predicted_set == gold_set:
            exact_match_count += 1

        tp += len(predicted_set & gold_set)
        fp += len(predicted_set - gold_set)
        fn += len(gold_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "total_samples": total_samples,
        "evaluation_samples": evaluation_samples,
        "parsed_samples": parsed_samples,
        "parse_success_rate": (parsed_samples / total_samples) if total_samples else 0.0,
        "nonempty_samples": nonempty_samples,
        "predicted_nonempty_rate": (nonempty_samples / total_samples) if total_samples else 0.0,
        "mean_predicted_relations": (predicted_relation_total / total_samples) if total_samples else 0.0,
        "exact_match_accuracy": (exact_match_count / evaluation_samples) if evaluation_samples else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "failure_counts": dict(sorted(failure_counts.items())),
        "allowed_relation_types": list(CANONICAL_LABELS),
    }


def format_metrics_report(metrics: Dict[str, Any], *, prediction_path: Optional[Path] = None) -> str:
    """Format aggregated metrics into a readable plain-text report."""

    lines: List[str] = []
    if prediction_path is not None:
        lines.append(f"Prediction file: {prediction_path}")
        lines.append("")

    lines.extend(
        [
            "Prediction Summary",
            "------------------",
            f"Total samples:           {metrics['total_samples']}",
            f"Evaluated samples:       {metrics['evaluation_samples']}",
            f"Parsed samples:          {metrics['parsed_samples']}",
            f"Parse success rate:      {metrics['parse_success_rate']:.3f}",
            f"Predicted non-empty:     {metrics['predicted_nonempty_rate']:.3f}",
            f"Mean predicted rels:     {metrics['mean_predicted_relations']:.3f}",
            "",
            "Extraction Metrics",
            "------------------",
            f"Exact match accuracy:    {metrics['exact_match_accuracy']:.3f}",
            f"Precision:               {metrics['precision']:.3f}",
            f"Recall:                  {metrics['recall']:.3f}",
            f"F1:                      {metrics['f1']:.3f}",
            f"TP / FP / FN:            {metrics['tp']} / {metrics['fp']} / {metrics['fn']}",
        ]
    )

    failure_counts = metrics.get("failure_counts") or {}
    if failure_counts:
        lines.extend(["", "Parse Failures", "--------------"])
        for reason, count in failure_counts.items():
            lines.append(f"{reason}: {count}")

    return "\n".join(lines)
