"""Quantitative evaluation summary script.

Reads two benchmark result sets (base model vs. fine-tuned LoRA model) and
generates comparison reports in plain text and Markdown formats.

Usage:
    python scripts/evaluation/quantitative_evaluation.py

Outputs:
    results/quantitative_report.txt
    results/quantitative_report.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_CSV = (
    PROJECT_ROOT
    / "results"
    / "benchmark_suite_latest_raw_clean_base_20260329"
    / "summary.csv"
)

LORA_CSV = (
    PROJECT_ROOT
    / "results"
    / "benchmark_suite_latest_raw_clean_balanced_e3_ckpt993_20260329"
    / "summary.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results"

DATASETS: List[str] = [
    "own_validation",
    "own_test",
    "seen_style_validation",
    "seen_style_test",
    "ade_corpus_v2",
    "phee_dev",
    "phee_test",
    "ddi2013_test",
    "tac2017_adr_gold",
    "cadec_meddra",
]

DATASET_LABELS: Dict[str, str] = {
    "own_validation": "Own Validation      (internal)",
    "own_test": "Own Test            (internal)",
    "seen_style_validation": "Seen-Style Val      (internal)",
    "seen_style_test": "Seen-Style Test     (internal)",
    "ade_corpus_v2": "ADE Corpus v2       (external)",
    "phee_dev": "PHEE Dev            (external)",
    "phee_test": "PHEE Test           (external)",
    "ddi2013_test": "DDI-2013 Test       (external)",
    "tac2017_adr_gold": "TAC-2017 ADR Gold   (external)",
    "cadec_meddra": "CADEC MedDRA        (external)",
}


def load_csv(path: Path) -> Dict[str, Dict[str, str]]:
    """Load a summary.csv file and return rows keyed by dataset name."""
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    data: Dict[str, Dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            data[row["dataset"]] = row
    return data


def pct(val: Optional[str]) -> str:
    """Format a float string as a percentage."""
    if not val or val.strip() == "":
        return "  --  "
    try:
        return f"{float(val) * 100:5.1f}%"
    except ValueError:
        return val


def avg_f1(data: Dict[str, Dict[str, str]], datasets: List[str]) -> float:
    """Compute the average F1 score across a dataset list."""
    values = [float(data.get(name, {}).get("f1", 0) or 0) for name in datasets]
    return sum(values) / len(values) if values else 0.0


def build_txt(base: Dict[str, Dict[str, str]], lora: Dict[str, Dict[str, str]]) -> str:
    """Build the plain-text comparison report."""
    internal = [name for name in DATASETS if "internal" in DATASET_LABELS[name]]
    external = [name for name in DATASETS if "external" in DATASET_LABELS[name]]
    sep = "-" * 100

    lines: List[str] = [
        "=" * 100,
        "COMP6713 DDI/ADE Extraction -- Quantitative Evaluation Report",
        "Base Model (Qwen3-8B, no fine-tuning)  vs.  LoRA Model (Qwen3-8B + rsLoRA, epoch-3, step 993)",
        "Metric: Micro Precision / Recall / F1  [triple-level exact match, case-insensitive]",
        "=" * 100,
        "",
        f"{'Dataset':<34} {'Base-P':>8} {'Base-R':>8} {'Base-F1':>8}"
        f"  |  {'LoRA-P':>8} {'LoRA-R':>8} {'LoRA-F1':>8}  {'dF1':>7}",
        sep,
        "[ Internal datasets -- same distribution as training data ]",
    ]

    for dataset in internal:
        base_row, lora_row = base.get(dataset, {}), lora.get(dataset, {})
        delta = float(lora_row.get("f1", 0) or 0) - float(base_row.get("f1", 0) or 0)
        lines.append(
            f"  {DATASET_LABELS[dataset]:<32} {pct(base_row.get('precision')):>8}"
            f" {pct(base_row.get('recall')):>8} {pct(base_row.get('f1')):>8}"
            f"  |  {pct(lora_row.get('precision')):>8} {pct(lora_row.get('recall')):>8}"
            f" {pct(lora_row.get('f1')):>8}  {delta * 100:+.1f}%"
        )

    lines += [sep, "[ External / held-out datasets -- never seen during training ]"]

    for dataset in external:
        base_row, lora_row = base.get(dataset, {}), lora.get(dataset, {})
        delta = float(lora_row.get("f1", 0) or 0) - float(base_row.get("f1", 0) or 0)
        lines.append(
            f"  {DATASET_LABELS[dataset]:<32} {pct(base_row.get('precision')):>8}"
            f" {pct(base_row.get('recall')):>8} {pct(base_row.get('f1')):>8}"
            f"  |  {pct(lora_row.get('precision')):>8} {pct(lora_row.get('recall')):>8}"
            f" {pct(lora_row.get('f1')):>8}  {delta * 100:+.1f}%"
        )

    avg_internal_base = avg_f1(base, internal)
    avg_internal_lora = avg_f1(lora, internal)
    avg_external_base = avg_f1(base, external)
    avg_external_lora = avg_f1(lora, external)

    lines += [
        sep,
        "",
        f"  {'Avg F1 -- Internal (4 datasets)':<32} {'':>26} {avg_internal_base * 100:>6.1f}%"
        f"  |  {'':>26} {avg_internal_lora * 100:>6.1f}%  {(avg_internal_lora - avg_internal_base) * 100:+.1f}%",
        f"  {'Avg F1 -- External (6 datasets)':<32} {'':>26} {avg_external_base * 100:>6.1f}%"
        f"  |  {'':>26} {avg_external_lora * 100:>6.1f}%  {(avg_external_lora - avg_external_base) * 100:+.1f}%",
        "",
        "Notes:",
        "  * Evaluation metric: triple-level exact match on (relation_type, head_entity, tail_entity),",
        "    with case-insensitive entity comparison.",
        "  * Base model: Qwen3-8B with no fine-tuning (zero-shot inference).",
        "  * LoRA model: Qwen3-8B fine-tuned with rsLoRA on balanced-augmented data, epoch-3 (step 993).",
        "  * DDI-2013: LoRA recall improves greatly but precision drops -- model over-predicts relations.",
        "  * TAC-2017 / CADEC: harder domain shift; both models struggle, but LoRA improves recall.",
        "=" * 100,
    ]
    return "\n".join(lines)


def build_md(base: Dict[str, Dict[str, str]], lora: Dict[str, Dict[str, str]]) -> str:
    """Build the Markdown comparison report."""
    internal = [name for name in DATASETS if "internal" in DATASET_LABELS[name]]
    external = [name for name in DATASETS if "external" in DATASET_LABELS[name]]

    def short_name(dataset: str) -> str:
        return DATASET_LABELS[dataset].split("(")[0].strip()

    def split_type(dataset: str) -> str:
        return "Internal" if "internal" in DATASET_LABELS[dataset] else "External"

    lines: List[str] = [
        "# Quantitative Evaluation",
        "",
        "Comparison of **Base model** (Qwen3-8B, no fine-tuning) vs. "
        "**LoRA model** (Qwen3-8B + rsLoRA, epoch-3, step 993).",
        "",
        "Metric: **Micro Precision / Recall / F1** -- triple-level exact match "
        "(`relation_type` + `head_entity` + `tail_entity`, case-insensitive).",
        "",
        "## Results",
        "",
        "| Dataset | Type | Base P | Base R | Base F1 | LoRA P | LoRA R | LoRA F1 | dF1 |",
        "|---------|------|-------:|-------:|--------:|-------:|-------:|--------:|----:|",
    ]

    for dataset in DATASETS:
        base_row, lora_row = base.get(dataset, {}), lora.get(dataset, {})
        delta = float(lora_row.get("f1", 0) or 0) - float(base_row.get("f1", 0) or 0)
        lines.append(
            f"| {short_name(dataset)} | {split_type(dataset)} "
            f"| {pct(base_row.get('precision'))} | {pct(base_row.get('recall'))} | {pct(base_row.get('f1'))} "
            f"| {pct(lora_row.get('precision'))} | {pct(lora_row.get('recall'))} | {pct(lora_row.get('f1'))} "
            f"| {delta * 100:+.1f}% |"
        )

    avg_internal_base = avg_f1(base, internal)
    avg_internal_lora = avg_f1(lora, internal)
    avg_external_base = avg_f1(base, external)
    avg_external_lora = avg_f1(lora, external)

    lines += [
        "",
        "## Summary",
        "",
        "| Split | Base Avg F1 | LoRA Avg F1 | Gain |",
        "|-------|------------:|------------:|-----:|",
        f"| Internal (4 datasets) | {avg_internal_base * 100:.1f}% | {avg_internal_lora * 100:.1f}% | {(avg_internal_lora - avg_internal_base) * 100:+.1f}% |",
        f"| External (6 datasets) | {avg_external_base * 100:.1f}% | {avg_external_lora * 100:.1f}% | {(avg_external_lora - avg_external_base) * 100:+.1f}% |",
        "",
        "## Notes",
        "",
        "- **External datasets** (ADE Corpus v2, PHEE, DDI-2013, TAC-2017 ADR, CADEC) were never seen during training.",
        "- **DDI-2013**: LoRA recall improves substantially but precision drops, indicating the model over-predicts relations.",
        "- **TAC-2017 / CADEC**: harder domain transfer; LoRA shows improved recall over the base model.",
        "- The base model occasionally fails to produce valid JSON output (parse_success_rate < 1.0 on some datasets).",
    ]
    return "\n".join(lines)


def main() -> None:
    """Load benchmark summaries and write both comparison reports."""
    base = load_csv(BASE_CSV)
    lora = load_csv(LORA_CSV)

    txt = build_txt(base, lora)
    md = build_md(base, lora)

    print(txt)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "quantitative_report.txt").write_text(txt, encoding="utf-8")
    (OUTPUT_DIR / "quantitative_report.md").write_text(md, encoding="utf-8")

    print("\nSaved -> results/quantitative_report.txt")
    print("Saved -> results/quantitative_report.md")


if __name__ == "__main__":
    main()
