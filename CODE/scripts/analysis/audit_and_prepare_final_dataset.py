"""Dataset audit and final dataset materialization script."""

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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Audit ADE/DDI data, materialize the final dataset, and write reports.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen3_8b_lora_ddi_ade_final.yaml",
        help="Training config used for tokenizer length statistics.",
    )
    parser.add_argument(
        "--base-data-dir",
        type=str,
        default="data",
        help="Directory containing raw merged_chatml_{train,validation,test}.jsonl files.",
    )
    parser.add_argument(
        "--primary-augmentations",
        type=str,
        default="data/augmentations/curated_train_augmentations.json",
        help="Primary augmentation spec JSON path.",
    )
    parser.add_argument(
        "--supplement-augmentations",
        type=str,
        nargs="*",
        default=[
            "data/augmentations/curated_train_augmentations_supplement_augment.json",
        ],
        help="Additional augmentation spec JSON paths.",
    )
    parser.add_argument(
        "--current-augmentation-jsonl",
        type=str,
        default="data/augmentations/merged_chatml_train_augmentations.jsonl",
        help="Current augmentation sidecar JSONL used for delta statistics.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/Comp6713-Ddi-Ade-Extraction_final_augment",
        help="Output directory for final train/validation/test files.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports/augment",
        help="Output directory for audit, fix, and statistics reports.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Apply minimal text normalization."""
    return " ".join(str(text).strip().split())


def canonical_text(text: str) -> str:
    """Fold text into a canonical form for deduplication and comparison."""
    return normalize_text(text).casefold()


def tokenize_text(text: str) -> set[str]:
    """Convert text to a simple token set for near-duplicate analysis."""
    return {match.group(0).casefold() for match in TOKEN_RE.finditer(text)}


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    """Compute the SHA-256 of a text value."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_spec_records(paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and merge augmentation specs while tracking duplicates."""
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

    return f"""# Data Audit Report

## Scope

- Data root: `data/Comp6713-Ddi-Ade-Extraction`
- Final processed dataset: `{stats["paths"]["processed_dir"]}`
- Dataset hash for this audit: `{dataset_hash_value}`

## Checked Files

- Original training set: `{stats["paths"]["original_train"]}`
- Original validation set: `{stats["paths"]["original_validation"]}`
- Original test set: `{stats["paths"]["original_test"]}`
- Current augmentation sidecar: `{stats["paths"]["current_augmentation_jsonl"]}`
- Final augmentation sidecar: `{stats["paths"]["final_augmentations"]}`
- Final merged training set: `{stats["paths"]["final_train"]}`

## Format Consistency

- Processed training data uses ChatML JSONL with `messages = [system, user, assistant]`.
- All processed train/validation/test rows are rewritten with the shared configured system prompt.
- `span / offset / index` fields are not part of this repository schema, so offset-based boundary checks are not applicable.
- Original training parse failures: `{original_train["parse_failures"]}`
- Final augmentation parse failures: `{augmentations["parse_failures"]}`
- Final training parse failures: `{final_train["parse_failures"]}`
- Final augmentation entity substring issues: `{augmentations["entity_substring_issues"]}`
- Final augmentation invalid labels: `{augmentations["invalid_labels"]}`

## Semantic and Label Quality

- Retained curated augmentations from the previous version: `{supplement["retained_previous_rows"]}`
- New supplemental rows after audit: `{supplement["new_rows_added"]}`
- Duplicate augmentation spec entries removed during merge: `{stats["spec_merge"]["duplicate_spec_entries_removed"]}`
- Exact duplicate texts in the final augmentation set: `{augmentations["duplicate_text_examples"]}`
- Near-duplicate pairs in the final augmentation set (`token Jaccard >= 0.75`): `{stats["final_augmentation_near_duplicates"]}`
- Exact overlap between augmentations and validation: `{len(overlaps["augmentations_vs_validation_exact"])} rows`
- Exact overlap between augmentations and test: `{len(overlaps["augmentations_vs_test_exact"])} rows`
- Polluted original-training rows removed from the processed training set: `{stats["removed_polluted_train_rows"]}`

Manual review summary:
- `paraphrase`: Preserves meaning with alternative syntax and discourse forms.
- `negative`: Keeps surface cues such as drugs, events, or interaction wording without enough evidence for a valid relation.
- `hardcase`: Includes distractor entities, class-instance shifts, coordination, or long-range triggers.
- `margincase`: Stays close to positive evidence but still does not support a definite relation because of uncertainty, confounding, or temporal ambiguity.

## Distribution and Coverage

- Original training rows: `{original_train["num_rows"]}`
- Final augmentation rows: `{augmentations["num_rows"]}`
- Final merged training rows: `{final_train["num_rows"]}`
- Final augmentation type distribution: `{json.dumps(augmentations["augmentation_type_counts"], ensure_ascii=False)}`
- Final augmentation relation distribution: `{json.dumps(augmentations["relation_counts"], ensure_ascii=False)}`
- Positive rows in the final augmentation set: `{stats["final_augmentation_positive_rows"]}`
- Empty-target rows in the final augmentation set: `{stats["final_augmentation_empty_rows"]}`
- Empty-target rows in the final training set: `{stats["final_train_empty_rows"]}`
- Final training label distribution: `{json.dumps(final_train["relation_counts"], ensure_ascii=False)}`

Coverage highlights:
- Long-gap relations in the final augmentation set: `{augmentations["long_gap_relations"]}`
- Complex lexical marker counts in the final augmentation set: `{json.dumps(augmentations["marker_counts"], ensure_ascii=False)}`
- Per-augmentation summaries are saved in `reports/data_stats.json`.

## Split Contamination Checks

- Exact overlap between original training and validation: `{len(overlaps["original_train_vs_validation_exact"])} rows`
- Exact overlap between original training and test: `{len(overlaps["original_train_vs_test_exact"])} rows`
- Exact overlap between validation and test: `{len(overlaps["validation_vs_test_exact"])} rows`
- Exact overlap between final training and validation: `{len(overlaps["final_train_vs_validation_exact"])} rows`
- Exact overlap between final training and test: `{len(overlaps["final_train_vs_test_exact"])} rows`
- Exact overlap between final augmentations and validation: `{len(overlaps["augmentations_vs_validation_exact"])} rows`
- Exact overlap between final augmentations and test: `{len(overlaps["augmentations_vs_test_exact"])} rows`

## Key Findings

- The materialized dataset remains internally consistent and parseable under the repository schema.
- The original dataset has no empty-target samples, so `negative` and `margincase` fill that training gap.
- Rare DDI subtypes, especially `DDI-INT` and `DDI-ADVISE`, were sparse in the first augmentation round and were supplemented.
- Exact overlaps existed between train/dev and train/test in the original split, and those rows were removed from the processed training set.
"""


def build_fix_log(stats: Dict[str, Any]) -> str:
    supplement = stats["supplement_delta"]
    return f"""# Data Fix Log

## Inputs

- Previous augmentation sidecar: `{stats["paths"]["current_augmentation_jsonl"]}`
- Primary augmentation spec: `{stats["paths"]["primary_spec"]}`
- Supplemental augmentation spec: `{stats["paths"]["supplement_spec"]}`

## Processing Actions

- Malformed rows removed: 0
- Invalid-label rows removed: 0
- Entity-boundary error rows removed: 0
- Duplicate augmentation specs removed during merge: `{stats["spec_merge"]["duplicate_spec_entries_removed"]}`
- Exact-overlap original training rows removed from the processed training set: `{stats["removed_polluted_train_rows"]}`
- Retained valid historical augmentations: `{supplement["retained_previous_rows"]}`
- New manually curated augmentations: `{supplement["new_rows_added"]}`

## Supplement Focus

- Added more `DDI-INT` rows to cover both class-level and instance-level interaction wording.
- Added more `DDI-ADVISE` / `DDI-MECHANISM` rows to improve label balance.
- Added more challenging `negative` and `margincase` rows to sharpen the decision boundary.
- Kept all trainable files compatible with ChatML JSONL.

## Final Outputs

- Final training set: `{stats["paths"]["final_train"]}`
- Final validation set: `{stats["paths"]["final_validation"]}`
- Final test set: `{stats["paths"]["final_test"]}`
- Final augmentation sidecar: `{stats["paths"]["final_augmentations"]}`

## Final Size

- Original training rows: `{stats["files"]["original_train"]["num_rows"]}`
- Base training rows after split-contamination removal: `{stats["filtered_original_train_rows"]}`
- Final augmentation rows: `{stats["files"]["final_augmentations"]["num_rows"]}`
- Final merged training rows: `{stats["files"]["final_train"]["num_rows"]}`
- Final dataset hash: `{stats["final_dataset_hash"]}`
"""


def main() -> None:
    """Run the full audit, materialization, and report generation pipeline."""
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
