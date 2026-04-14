from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_backends import generate_predictions, load_model_and_tokenizer_vllm
from src.inference_config import load_inference_config
from src.parser import DatasetExample, evaluate_prediction_rows, format_metrics_report, load_dataset_examples
from src.prompting import load_system_prompt


VARIANT_SPECS: Dict[str, Dict[str, Optional[str]]] = {
    "base": {
        "adapter_path": None,
        "label": "base",
    },
    "lora": {
        "adapter_path": "outputs/qwen3_8b_lora_ddi_ade_c5fc8c06/final_adapter",
        "label": "base+lora",
    },
    "rslora_620": {
        "adapter_path": "outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/checkpoint-620",
        "label": "base+rslora@620",
    },
    "rslora_930": {
        "adapter_path": "outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/checkpoint-930",
        "label": "base+rslora@930",
    },
}

DEFAULT_OWN_VALIDATION_PATH = "data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/merged_chatml_validation.jsonl"
DEFAULT_OWN_TEST_PATH = "data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/merged_chatml_test.jsonl"

METRIC_MODE_LABELED = "labeled_task"
METRIC_MODE_EMPTY_GUARDRAIL = "empty_guardrail"
METRIC_MODE_UNLABELED_SCHEMA = "unlabeled_schema"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full vLLM benchmark suite serially.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "lora", "rslora_620", "rslora_930"],
        help="Preset variant ids to run.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/benchmark_suite_vllm_batch64",
        help="Output directory for predictions, metrics, and summary CSV files.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of generated tokens.")
    parser.add_argument(
        "--only-datasets",
        nargs="*",
        default=None,
        help="Optional dataset allowlist.",
    )
    parser.add_argument(
        "--skip-datasets",
        nargs="*",
        default=None,
        help="Optional dataset blocklist.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Optional per-dataset sample cap for smoke runs.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Run a single custom adapter path instead of the preset variants.",
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default="custom",
        help="Variant id used together with --adapter-path.",
    )
    parser.add_argument(
        "--variant-label",
        type=str,
        default=None,
        help="Human-readable label used together with --adapter-path.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help="Override the base model path.",
    )
    parser.add_argument(
        "--own-validation-path",
        type=str,
        default=DEFAULT_OWN_VALIDATION_PATH,
        help="Repository validation-set path; defaults to the latest_raw_clean split.",
    )
    parser.add_argument(
        "--own-test-path",
        type=str,
        default=DEFAULT_OWN_TEST_PATH,
        help="Repository test-set path; defaults to the latest_raw_clean split.",
    )
    return parser.parse_args()


def resolve_project_or_absolute_path(raw_path: str | Path) -> Path:
    """Resolve a command-line path to an absolute path."""
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def normalize_text(value: str) -> str:
    return " ".join(str(value).split())


def iter_strings(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_strings(item)
        return
    if isinstance(value, dict):
        text_value = value.get("text")
        if text_value is not None:
            yield from iter_strings(text_value)
        return


def unique_preserving_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        normalized = normalize_text(item)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def split_text_into_chunks(text: str, *, max_chars: int) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    sentence_delimiters = ".!?;。！？；"
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    sentences: List[str] = []
    sentence_buffer: List[str] = []
    for char in normalized:
        sentence_buffer.append(char)
        if char in sentence_delimiters:
            sentence = "".join(sentence_buffer).strip()
            if sentence:
                sentences.append(sentence)
            sentence_buffer = []
    if sentence_buffer:
        tail = "".join(sentence_buffer).strip()
        if tail:
            sentences.append(tail)
    if not sentences:
        sentences = [normalized]

    def flush_current() -> None:
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current).strip())
            current = []
            current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            flush_current()
            for start in range(0, len(sentence), max_chars):
                piece = sentence[start : start + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue
        projected_len = current_len + len(sentence) + (1 if current else 0)
        if current and projected_len > max_chars:
            flush_current()
        current.append(sentence)
        current_len += len(sentence) + (1 if current_len else 0)

    flush_current()
    return chunks or [normalized[:max_chars]]


def chunk_sentences(sentences: Sequence[str], *, max_chars: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        normalized = normalize_text(sentence)
        if not normalized:
            continue
        if len(normalized) > max_chars:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            chunks.extend(split_text_into_chunks(normalized, max_chars=max_chars))
            continue

        projected_len = current_len + len(normalized) + (1 if current else 0)
        if current and projected_len > max_chars:
            chunks.append(" ".join(current).strip())
            current = []
            current_len = 0

        current.append(normalized)
        current_len += len(normalized) + (1 if current_len else 0)

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def chunk_raw_text_with_offsets(text: str, *, max_chars: int) -> List[Tuple[str, int, int]]:
    raw_text = str(text or "")
    if not raw_text.strip():
        return []

    length = len(raw_text)
    chunks: List[Tuple[str, int, int]] = []
    start = 0
    delimiters = ".!?;\n。！？；"

    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            while end > start and raw_text[end - 1] not in delimiters:
                end -= 1
            if end == start:
                end = min(start + max_chars, length)

        chunk_text = raw_text[start:end].strip()
        if chunk_text:
            chunks.append((chunk_text, start, end))
        start = end

    return chunks


def parse_offset_list(raw_value: str) -> List[int]:
    offsets: List[int] = []
    for part in str(raw_value or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            offsets.append(int(part))
        except ValueError:
            continue
    return offsets


def build_example(
    *,
    sample_id: str,
    split: str,
    system_prompt: str,
    user_text: str,
    gold_relations: Sequence[Dict[str, str]],
) -> DatasetExample:
    return DatasetExample(
        sample_id=sample_id,
        split=split,
        system_prompt=system_prompt,
        user_text=user_text,
        gold_relations=list(gold_relations),
    )


def build_chatml_dataset(path: Path, *, split: str, limit: Optional[int] = None) -> List[DatasetExample]:
    return load_dataset_examples(path, split=split, limit=limit)


def build_ade_corpus_v2_dataset(
    path: Path,
    *,
    split: str,
    system_prompt: str,
    limit: Optional[int] = None,
) -> List[DatasetExample]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[row["text"]].append(
                {
                    "head_entity": normalize_text(row["drug"]),
                    "tail_entity": normalize_text(row["effect"]),
                    "relation_type": "ADE",
                }
            )

    examples: List[DatasetExample] = []
    for index, (text, relations) in enumerate(grouped.items()):
        examples.append(
            build_example(
                sample_id=f"{split}_{index:05d}",
                split=split,
                system_prompt=system_prompt,
                user_text=text,
                gold_relations=relations,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def build_phee_dataset(
    path: Path,
    *,
    split: str,
    system_prompt: str,
    limit: Optional[int] = None,
) -> List[DatasetExample]:
    examples: List[DatasetExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            relations: List[Dict[str, str]] = []
            for annotation in row.get("annotations", []):
                for event in annotation.get("events", []):
                    if event.get("event_type") != "Adverse_event":
                        continue
                    treatment = event.get("Treatment", {})
                    effect = event.get("Effect", {})
                    drugs = unique_preserving_order(iter_strings(treatment.get("Drug", {}).get("text")))
                    effects = unique_preserving_order(iter_strings(effect.get("text")))
                    for drug in drugs:
                        for adverse_effect in effects:
                            relations.append(
                                {
                                    "head_entity": drug,
                                    "tail_entity": adverse_effect,
                                    "relation_type": "ADE",
                                }
                            )
            examples.append(
                build_example(
                    sample_id=str(row.get("id") or f"{split}_{index:05d}"),
                    split=split,
                    system_prompt=system_prompt,
                    user_text=str(row.get("context", "")),
                    gold_relations=relations,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def build_ddi2013_dataset(
    xml_root: Path,
    *,
    split: str,
    system_prompt: str,
    limit: Optional[int] = None,
) -> List[DatasetExample]:
    label_map = {
        "mechanism": "DDI-MECHANISM",
        "effect": "DDI-EFFECT",
        "advise": "DDI-ADVISE",
        "int": "DDI-INT",
    }
    examples: List[DatasetExample] = []
    for xml_path in sorted(xml_root.rglob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for sentence in root.findall("sentence"):
            entities = {entity.get("id"): entity.get("text", "") for entity in sentence.findall("entity")}
            relations: List[Dict[str, str]] = []
            for pair in sentence.findall("pair"):
                if pair.get("ddi") != "true":
                    continue
                relation_type = label_map.get((pair.get("type") or "").lower())
                if relation_type is None:
                    continue
                relations.append(
                    {
                        "head_entity": normalize_text(entities.get(pair.get("e1"), "")),
                        "tail_entity": normalize_text(entities.get(pair.get("e2"), "")),
                        "relation_type": relation_type,
                    }
                )
            examples.append(
                build_example(
                    sample_id=str(sentence.get("id") or f"{split}_{len(examples):05d}"),
                    split=split,
                    system_prompt=system_prompt,
                    user_text=str(sentence.get("text", "")),
                    gold_relations=relations,
                )
            )
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


def build_tac2017_adr_dataset(
    xml_root: Path,
    *,
    split: str,
    system_prompt: str,
    limit: Optional[int] = None,
) -> List[DatasetExample]:
    examples: List[DatasetExample] = []
    for xml_path in sorted(xml_root.rglob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        drug_name = normalize_text(root.attrib.get("drug", ""))

        sections_by_id: Dict[str, str] = {}
        text_root = root.find("Text")
        if text_root is not None:
            for section in text_root.findall("Section"):
                section_id = str(section.attrib.get("id") or section.attrib.get("name") or "")
                section_text = "".join(section.itertext())
                if section_id and section_text:
                    sections_by_id[section_id] = section_text

        section_reactions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        mentions_root = root.find("Mentions")
        if mentions_root is not None:
            for mention in mentions_root.findall("Mention"):
                if mention.attrib.get("type") != "AdverseReaction":
                    continue
                section_id = str(mention.attrib.get("section") or "")
                reaction_text = normalize_text(mention.attrib.get("str", ""))
                if not section_id or not reaction_text:
                    continue
                section_reactions[section_id].append(
                    {
                        "starts": parse_offset_list(mention.attrib.get("start", "")),
                        "lens": parse_offset_list(mention.attrib.get("len", "")),
                        "head_entity": drug_name,
                        "tail_entity": reaction_text,
                        "relation_type": "ADE",
                    }
                )

        for section_id, raw_text in sections_by_id.items():
            if not raw_text:
                continue

            for chunk_index, (chunk_text, chunk_start, chunk_end) in enumerate(
                chunk_raw_text_with_offsets(raw_text, max_chars=3500)
            ):
                gold_relations = [
                    {
                        "head_entity": relation["head_entity"],
                        "tail_entity": relation["tail_entity"],
                        "relation_type": relation["relation_type"],
                    }
                    for relation in section_reactions.get(section_id, [])
                    if any(chunk_start <= start < chunk_end for start in relation.get("starts", []))
                ]
                examples.append(
                    build_example(
                        sample_id=f"{xml_path.stem}_{section_id}_chunk{chunk_index:02d}",
                        split=split,
                        system_prompt=system_prompt,
                        user_text=normalize_text(chunk_text),
                        gold_relations=gold_relations,
                    )
                )
                if limit is not None and len(examples) >= limit:
                    return examples
    return examples


def build_cadec_dataset(
    text_root: Path,
    ann_root: Path,
    *,
    split: str,
    system_prompt: str,
    limit: Optional[int] = None,
) -> List[DatasetExample]:
    examples: List[DatasetExample] = []
    for text_path in sorted(text_root.glob("*.txt")):
        stem = text_path.stem
        ann_path = ann_root / f"{stem}.ann"
        if not ann_path.exists():
            continue
        drug_name = normalize_text(stem.split(".")[0].replace("_", " "))
        relations: List[Dict[str, str]] = []
        with ann_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or not line.startswith("TT"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                reaction_text = normalize_text(parts[2])
                if not reaction_text:
                    continue
                relations.append(
                    {
                        "head_entity": drug_name,
                        "tail_entity": reaction_text,
                        "relation_type": "ADE",
                    }
                )
        examples.append(
            build_example(
                sample_id=f"{split}_{stem}",
                split=split,
                system_prompt=system_prompt,
                user_text=text_path.read_text(encoding="utf-8"),
                gold_relations=relations,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def build_ifeval_dataset(
    path: Path,
    *,
    split: str,
    system_prompt: str,
    limit: Optional[int] = None,
) -> List[DatasetExample]:
    examples: List[DatasetExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                build_example(
                    sample_id=f"{split}_{row.get('key', index)}",
                    split=split,
                    system_prompt=system_prompt,
                    user_text=str(row.get("prompt", "")),
                    gold_relations=[],
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def build_longbench_dataset(
    path: Path,
    *,
    split: str,
    system_prompt: str,
    limit: Optional[int] = None,
) -> List[DatasetExample]:
    examples: List[DatasetExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            segments = []
            context = normalize_text(str(row.get("context", "")))
            question = normalize_text(str(row.get("input", "")))
            if context:
                segments.append(context)
            if question:
                segments.append(f"Question: {question}")
            user_text = "\n\n".join(segments).strip()
            for chunk_index, chunk in enumerate(split_text_into_chunks(user_text, max_chars=4000)):
                examples.append(
                    build_example(
                        sample_id=f"{split}_{index:05d}_chunk{chunk_index:02d}",
                        split=split,
                        system_prompt=system_prompt,
                        user_text=chunk,
                        gold_relations=[],
                    )
                )
                if limit is not None and len(examples) >= limit:
                    return examples
    return examples


def build_docred_dataset(
    path: Path,
    *,
    split: str,
    system_prompt: str,
    limit: Optional[int] = None,
) -> List[DatasetExample]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = json.load(handle)
    examples: List[DatasetExample] = []
    for index, row in enumerate(rows):
        title = normalize_text(str(row.get("title", "")))
        raw_sentences = [" ".join(tokens) for tokens in row.get("sents", [])]
        sentence_chunks = chunk_sentences(raw_sentences, max_chars=3800)
        if not sentence_chunks:
            sentence_chunks = [title] if title else []
        for chunk_index, chunk in enumerate(sentence_chunks):
            user_text = "\n\n".join(filter(None, [title, chunk]))
            examples.append(
                build_example(
                    sample_id=f"{split}_{index:05d}_chunk{chunk_index:02d}",
                    split=split,
                    system_prompt=system_prompt,
                    user_text=user_text,
                    gold_relations=[],
                )
            )
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


def build_tac2018_schema_dataset(
    xml_root: Path,
    *,
    split: str,
    system_prompt: str,
    limit: Optional[int] = None,
) -> List[DatasetExample]:
    examples: List[DatasetExample] = []
    for xml_path in sorted(xml_root.rglob("*.xml")):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        sentences_root = root.find("Sentences")
        section_sentences: Dict[str, List[str]] = defaultdict(list)
        if sentences_root is not None:
            for sentence in sentences_root.findall("Sentence"):
                section_id = str(sentence.attrib.get("section") or "")
                sentence_text = normalize_text("".join(sentence.itertext()))
                if section_id and sentence_text:
                    section_sentences[section_id].append(sentence_text)

        text_root = root.find("Text")
        if text_root is not None:
            for section in text_root.findall("Section"):
                section_id = str(section.attrib.get("id") or "")
                section_name = normalize_text(str(section.attrib.get("name") or section_id or "section"))
                sentence_chunks = chunk_sentences(section_sentences.get(section_id, []), max_chars=3800)
                if not sentence_chunks:
                    section_text = normalize_text("".join(section.itertext()))
                    sentence_chunks = split_text_into_chunks(section_text, max_chars=3800)

                for chunk_index, chunk in enumerate(sentence_chunks):
                    section_prefix = f"Section: {section_name}"
                    user_text = "\n\n".join([section_prefix, chunk]).strip()
                    examples.append(
                        build_example(
                            sample_id=f"{split}_{xml_path.stem}_{section_id}_chunk{chunk_index:02d}",
                            split=split,
                            system_prompt=system_prompt,
                            user_text=user_text,
                            gold_relations=[],
                        )
                    )
                    if limit is not None and len(examples) >= limit:
                        return examples
    return examples


def write_prediction_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_rows = len(rows)
    parsed_rows = sum(1 for row in rows if row.get("parse_status") == "parsed")
    nonempty_rows = sum(1 for row in rows if row.get("predicted_relations"))
    relation_total = sum(len(row.get("predicted_relations", [])) for row in rows)
    return {
        "total_samples": total_rows,
        "parsed_samples": parsed_rows,
        "parse_success_rate": (parsed_rows / total_rows) if total_rows else 0.0,
        "predicted_nonempty_rate": (nonempty_rows / total_rows) if total_rows else 0.0,
        "mean_predicted_relations": (relation_total / total_rows) if total_rows else 0.0,
    }


def dataset_specs(
    system_prompt: str,
    limit: Optional[int],
    *,
    own_validation_path: Path,
    own_test_path: Path,
) -> List[Dict[str, Any]]:
    eval_root = PROJECT_ROOT / "evaluate_datasets"
    return [
        {
            "name": "own_validation",
            "group": "own_splits",
            "metric_mode": METRIC_MODE_LABELED,
            "builder": lambda: build_chatml_dataset(
                own_validation_path,
                split="own_validation",
                limit=limit,
            ),
        },
        {
            "name": "own_test",
            "group": "own_splits",
            "metric_mode": METRIC_MODE_LABELED,
            "builder": lambda: build_chatml_dataset(
                own_test_path,
                split="own_test",
                limit=limit,
            ),
        },
        {
            "name": "seen_style_validation",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_LABELED,
            "builder": lambda: build_chatml_dataset(
                eval_root / "seen_style_core/official_held_out/merged_chatml_validation.jsonl",
                split="seen_style_val",
                limit=limit,
            ),
        },
        {
            "name": "seen_style_test",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_LABELED,
            "builder": lambda: build_chatml_dataset(
                eval_root / "seen_style_core/official_held_out/merged_chatml_test.jsonl",
                split="seen_style_test",
                limit=limit,
            ),
        },
        {
            "name": "ade_corpus_v2",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_LABELED,
            "builder": lambda: build_ade_corpus_v2_dataset(
                eval_root / "ade_transfer/ADE_Corpus_V2/processed/drug_ade_relation.jsonl",
                split="ade_corpus_v2",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "phee_dev",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_LABELED,
            "builder": lambda: build_phee_dataset(
                eval_root / "ade_transfer/PHEE/raw/dev.json",
                split="phee_dev",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "phee_test",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_LABELED,
            "builder": lambda: build_phee_dataset(
                eval_root / "ade_transfer/PHEE/raw/test.json",
                split="phee_test",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "ddi2013_test",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_LABELED,
            "builder": lambda: build_ddi2013_dataset(
                eval_root / "ddi_transfer/DDIExtraction2013/extracted/DDICorpus/Test/Test for DDI Extraction task",
                split="ddi2013_test",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "tac2018_test1",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_UNLABELED_SCHEMA,
            "builder": lambda: build_tac2018_schema_dataset(
                eval_root / "ddi_transfer/TAC2018_DDI/extracted/test1Files",
                split="tac2018_test1",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "tac2018_test2",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_UNLABELED_SCHEMA,
            "builder": lambda: build_tac2018_schema_dataset(
                eval_root / "ddi_transfer/TAC2018_DDI/extracted/test2Files",
                split="tac2018_test2",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "tac2017_adr_gold",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_LABELED,
            "builder": lambda: build_tac2017_adr_dataset(
                eval_root / "pharmacovigilance_cross_genre/TAC2017_ADR/extracted/gold_xml",
                split="tac2017_adr",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "cadec_meddra",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_LABELED,
            "builder": lambda: build_cadec_dataset(
                eval_root / "pharmacovigilance_cross_genre/CADEC/extracted/cadec/text",
                eval_root / "pharmacovigilance_cross_genre/CADEC/extracted/cadec/meddra",
                split="cadec_meddra",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "ifeval_input",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_EMPTY_GUARDRAIL,
            "builder": lambda: build_ifeval_dataset(
                eval_root / "general_guardrails/IFEval/raw/ifeval_input_data.jsonl",
                split="ifeval",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "docred_dev",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_EMPTY_GUARDRAIL,
            "builder": lambda: build_docred_dataset(
                eval_root / "general_guardrails/DocRED/raw/dev.json.gz",
                split="docred_dev",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "docred_test",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_EMPTY_GUARDRAIL,
            "builder": lambda: build_docred_dataset(
                eval_root / "general_guardrails/DocRED/raw/test.json.gz",
                split="docred_test",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "longbench_multifieldqa_en",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_EMPTY_GUARDRAIL,
            "builder": lambda: build_longbench_dataset(
                eval_root / "general_guardrails/LongBench/light/multifieldqa_en.jsonl",
                split="longbench_multifieldqa_en",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "longbench_multifieldqa_zh",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_EMPTY_GUARDRAIL,
            "builder": lambda: build_longbench_dataset(
                eval_root / "general_guardrails/LongBench/light/multifieldqa_zh.jsonl",
                split="longbench_multifieldqa_zh",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "longbench_passage_retrieval_en",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_EMPTY_GUARDRAIL,
            "builder": lambda: build_longbench_dataset(
                eval_root / "general_guardrails/LongBench/light/passage_retrieval_en.jsonl",
                split="longbench_passage_retrieval_en",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "longbench_passage_retrieval_zh",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_EMPTY_GUARDRAIL,
            "builder": lambda: build_longbench_dataset(
                eval_root / "general_guardrails/LongBench/light/passage_retrieval_zh.jsonl",
                split="longbench_passage_retrieval_zh",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "longbench_gov_report",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_EMPTY_GUARDRAIL,
            "builder": lambda: build_longbench_dataset(
                eval_root / "general_guardrails/LongBench/light/gov_report.jsonl",
                split="longbench_gov_report",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
        {
            "name": "longbench_vcsum",
            "group": "evaluate_datasets",
            "metric_mode": METRIC_MODE_EMPTY_GUARDRAIL,
            "builder": lambda: build_longbench_dataset(
                eval_root / "general_guardrails/LongBench/light/vcsum.jsonl",
                split="longbench_vcsum",
                system_prompt=system_prompt,
                limit=limit,
            ),
        },
    ]


def build_variant_config(
    *,
    adapter_path: Optional[Path],
    batch_size: int,
    max_new_tokens: int,
    base_model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    config = load_inference_config("configs/infer_qwen3_8b_lora_ddi_ade_final.yaml", validate=False)
    config["backend"] = "vllm"
    config["model"]["base_model_name_or_path"] = (
        base_model_path if base_model_path is not None else PROJECT_ROOT / "models" / "Qwen3-8B"
    )
    config["model"]["adapter_path"] = adapter_path
    config["inference"]["batch_size"] = batch_size
    config["inference"]["max_new_tokens"] = max_new_tokens
    config["output"] = {}
    return config


def resolve_variant_entries(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Normalize preset or custom adapters into benchmark run entries."""
    if args.adapter_path is not None:
        return [
            {
                "name": str(args.variant_name or "custom"),
                "label": str(args.variant_label or args.variant_name or "custom"),
                "adapter_path": resolve_project_or_absolute_path(args.adapter_path),
            }
        ]

    entries: List[Dict[str, Any]] = []
    for variant in args.variants:
        spec = VARIANT_SPECS.get(variant)
        if spec is None:
            raise ValueError(f"Unknown variant: {variant}")
        adapter_path = spec["adapter_path"]
        entries.append(
            {
                "name": variant,
                "label": spec["label"],
                "adapter_path": (PROJECT_ROOT / adapter_path).resolve() if adapter_path is not None else None,
            }
        )
    return entries


def ensure_variant_paths(variant_entries: Sequence[Dict[str, Any]]) -> None:
    for variant in variant_entries:
        adapter_path = variant["adapter_path"]
        if adapter_path is None:
            continue
        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"Adapter not found for variant {variant['name']}: {adapter_path}")


def should_include_dataset(name: str, *, only: Optional[Sequence[str]], skip: Optional[Sequence[str]]) -> bool:
    if only and name not in set(only):
        return False
    if skip and name in set(skip):
        return False
    return True


def main() -> None:
    args = parse_args()
    variant_entries = resolve_variant_entries(args)
    ensure_variant_paths(variant_entries)

    system_prompt = load_system_prompt(str(PROJECT_ROOT / "prompts" / "medical_relation_extraction_system_prompt.txt"))
    own_validation_path = resolve_project_or_absolute_path(args.own_validation_path)
    own_test_path = resolve_project_or_absolute_path(args.own_test_path)
    specs = [
        spec
        for spec in dataset_specs(
            system_prompt=system_prompt,
            limit=args.limit_per_dataset,
            own_validation_path=own_validation_path,
            own_test_path=own_test_path,
        )
        if should_include_dataset(spec["name"], only=args.only_datasets, skip=args.skip_datasets)
    ]
    if not specs:
        raise ValueError("No datasets selected.")

    results_root = PROJECT_ROOT / args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)
    summary_path = results_root / "summary.csv"
    base_model_path = resolve_project_or_absolute_path(args.base_model_path) if args.base_model_path else None

    for variant in variant_entries:
        variant_dir = results_root / variant["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        config = build_variant_config(
            adapter_path=variant["adapter_path"],
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            base_model_path=base_model_path,
        )

        print(f"[{variant['name']}] building datasets...")
        built_specs: List[Tuple[Dict[str, Any], List[DatasetExample]]] = []
        for spec in specs:
            examples = spec["builder"]()
            built_specs.append((spec, examples))
            print(
                f"[{variant['name']}] dataset {spec['name']} ({spec['metric_mode']}): {len(examples)} samples",
                flush=True,
            )

        print(f"[{variant['name']}] loading model with vLLM...", flush=True)
        llm, tokenizer, sampling_params_class, lora_request_class = load_model_and_tokenizer_vllm(config)

        summary_rows: List[Dict[str, Any]] = []
        try:
            for spec, examples in built_specs:
                dataset_name = spec["name"]
                metric_mode = spec["metric_mode"]
                print(f"[{variant['name']}] running {dataset_name} ({len(examples)} samples)...", flush=True)
                rows = generate_predictions((llm, sampling_params_class, lora_request_class), tokenizer, examples, config)

                pred_path = variant_dir / f"{dataset_name}_predictions.jsonl"
                write_prediction_rows(pred_path, rows)

                base_summary = {
                    "variant": variant["name"],
                    "variant_label": variant["label"],
                    "dataset_group": spec["group"],
                    "dataset": dataset_name,
                    "metric_mode": metric_mode,
                }

                if metric_mode == METRIC_MODE_UNLABELED_SCHEMA:
                    metrics = summarize_rows(rows)
                    metrics_json_path = variant_dir / f"{dataset_name}_schema_metrics.json"
                    metrics_json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                    summary_row = {
                        **base_summary,
                        **metrics,
                        "exact_match_accuracy": "",
                        "precision": "",
                        "recall": "",
                        "f1": "",
                        "tp": "",
                        "fp": "",
                        "fn": "",
                    }
                    print(
                        f"[{variant['name']}] done {dataset_name}: parse={metrics['parse_success_rate']:.4f} "
                        f"nonempty={metrics['predicted_nonempty_rate']:.4f}",
                        flush=True,
                    )
                else:
                    metrics = evaluate_prediction_rows(rows)
                    metrics_json_path = variant_dir / f"{dataset_name}_metrics.json"
                    metrics_txt_path = variant_dir / f"{dataset_name}_metrics.txt"
                    metrics_json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                    metrics_txt_path.write_text(
                        format_metrics_report(metrics, prediction_path=pred_path),
                        encoding="utf-8",
                    )
                    schema_metrics = summarize_rows(rows)
                    summary_row = {
                        **base_summary,
                        "total_samples": metrics["total_samples"],
                        "parsed_samples": metrics["parsed_samples"],
                        "parse_success_rate": metrics["parse_success_rate"],
                        "predicted_nonempty_rate": schema_metrics["predicted_nonempty_rate"],
                        "mean_predicted_relations": schema_metrics["mean_predicted_relations"],
                        "exact_match_accuracy": metrics["exact_match_accuracy"],
                        "precision": metrics["micro"]["precision"],
                        "recall": metrics["micro"]["recall"],
                        "f1": metrics["micro"]["f1"],
                        "tp": metrics["micro"]["tp"],
                        "fp": metrics["micro"]["fp"],
                        "fn": metrics["micro"]["fn"],
                    }
                    print(
                        f"[{variant['name']}] done {dataset_name}: "
                        f"P={metrics['micro']['precision']:.4f} "
                        f"R={metrics['micro']['recall']:.4f} "
                        f"F1={metrics['micro']['f1']:.4f} "
                        f"EM={metrics['exact_match_accuracy']:.4f}",
                        flush=True,
                    )

                summary_rows.append(summary_row)
        finally:
            try:
                del llm
                del tokenizer
            except Exception:
                pass
            gc.collect()

        mode = "a" if summary_path.exists() else "w"
        with summary_path.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            if mode == "w":
                writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[{variant['name']}] summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
