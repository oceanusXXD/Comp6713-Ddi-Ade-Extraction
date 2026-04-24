"""
Quantitative Evaluation Summary Script
COMP6713 DDI/ADE Extraction Project

Reads two sets of benchmark results (Base model vs. fine-tuned LoRA model)
and generates a comparison report in plain text and Markdown formats.

Usage (run from the project root):
    python scripts/evaluation/quantitative_evaluation.py

Outputs:
    ../MISC/results/quantitative_report.txt  -- plain-text report (terminal-friendly)
    ../MISC/results/quantitative_report.md   -- Markdown report (can be pasted into README or report)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

# -- Path configuration -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MISC_ROOT = PROJECT_ROOT.parent / "MISC"

# Base model benchmark results (Qwen3-8B, no fine-tuning)
BASE_CSV = (
    MISC_ROOT
    / "results"
    / "benchmark_suite_latest_raw_clean_base_20260329"
    / "summary.csv"
)

# LoRA model benchmark results (Qwen3-8B + rsLoRA, epoch-3, step 993)
LORA_CSV = (
    MISC_ROOT
    / "results"
    / "benchmark_suite_latest_raw_clean_balanced_e3_ckpt993_20260329"
    / "summary.csv"
)

OUTPUT_DIR = MISC_ROOT / "results"

# -- Dataset display order and human-readable labels --------------------------

# Only labeled_task datasets are included; guardrail/unlabeled datasets are excluded.
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
    "own_validation":        "Own Validation      (internal)",
    "own_test":              "Own Test            (internal)",
    "seen_style_validation": "Seen-Style Val      (internal)",
    "seen_style_test":       "Seen-Style Test     (internal)",
    "ade_corpus_v2":         "ADE Corpus v2       (external)",
    "phee_dev":              "PHEE Dev            (external)",
    "phee_test":             "PHEE Test           (external)",
    "ddi2013_test":          "DDI-2013 Test       (external)",
    "tac2017_adr_gold":      "TAC-2017 ADR Gold   (external)",
    "cadec_meddra":          "CADEC MedDRA        (external)",
}


# -- CSV loading --------------------------------------------------------------

def load_csv(path: Path) -> Dict[str, Dict]:
    """Load a summary.csv file and return a dict keyed by dataset name."""
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    data: Dict[str, Dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["dataset"]] = row
    return data


# -- Formatting helpers -------------------------------------------------------

def pct(val: Optional[str]) -> str:
    """Format a float string as a percentage; return '  --  ' for missing values."""
    if not val or val.strip() == "":
        return "  --  "
    try:
        return f"{float(val) * 100:5.1f}%"
    except ValueError:
        return val


def avg_f1(data: Dict, ds_list: List[str]) -> float:
    """Compute the average F1 score across a list of datasets."""
    vals = [float(data.get(d, {}).get("f1", 0) or 0) for d in ds_list]
    return sum(vals) / len(vals) if vals else 0.0


# -- Plain-text report --------------------------------------------------------

def build_txt(base: Dict, lora: Dict) -> str:
    """Build a plain-text comparison report."""
    int_ds = [d for d in DATASETS if "internal" in DATASET_LABELS[d]]
    ext_ds = [d for d in DATASETS if "external" in DATASET_LABELS[d]]
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

    for ds in int_ds:
        b, l = base.get(ds, {}), lora.get(ds, {})
        delta = float(l.get("f1", 0) or 0) - float(b.get("f1", 0) or 0)
        lines.append(
            f"  {DATASET_LABELS[ds]:<32} {pct(b.get('precision')):>8}"
            f" {pct(b.get('recall')):>8} {pct(b.get('f1')):>8}"
            f"  |  {pct(l.get('precision')):>8} {pct(l.get('recall')):>8}"
            f" {pct(l.get('f1')):>8}  {delta*100:+.1f}%"
        )

    lines += [sep, "[ External / held-out datasets -- never seen during training ]"]

    for ds in ext_ds:
        b, l = base.get(ds, {}), lora.get(ds, {})
        delta = float(l.get("f1", 0) or 0) - float(b.get("f1", 0) or 0)
        lines.append(
            f"  {DATASET_LABELS[ds]:<32} {pct(b.get('precision')):>8}"
            f" {pct(b.get('recall')):>8} {pct(b.get('f1')):>8}"
            f"  |  {pct(l.get('precision')):>8} {pct(l.get('recall')):>8}"
            f" {pct(l.get('f1')):>8}  {delta*100:+.1f}%"
        )

    ai = avg_f1(base, int_ds)
    li = avg_f1(lora, int_ds)
    ae = avg_f1(base, ext_ds)
    le = avg_f1(lora, ext_ds)

    lines += [
        sep,
        "",
        f"  {'Avg F1 -- Internal (4 datasets)':<32} {'':>26} {ai*100:>6.1f}%"
        f"  |  {'':>26} {li*100:>6.1f}%  {(li-ai)*100:+.1f}%",
        f"  {'Avg F1 -- External (6 datasets)':<32} {'':>26} {ae*100:>6.1f}%"
        f"  |  {'':>26} {le*100:>6.1f}%  {(le-ae)*100:+.1f}%",
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


# -- Markdown report ----------------------------------------------------------

def build_md(base: Dict, lora: Dict) -> str:
    """Build a Markdown comparison report."""
    int_ds = [d for d in DATASETS if "internal" in DATASET_LABELS[d]]
    ext_ds = [d for d in DATASETS if "external" in DATASET_LABELS[d]]

    def short(ds: str) -> str:
        """Return the dataset name without the (internal/external) suffix."""
        return DATASET_LABELS[ds].split("(")[0].strip()

    def ds_type(ds: str) -> str:
        return "Internal" if "internal" in DATASET_LABELS[ds] else "External"

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

    for ds in DATASETS:
        b, l = base.get(ds, {}), lora.get(ds, {})
        delta = float(l.get("f1", 0) or 0) - float(b.get("f1", 0) or 0)
        lines.append(
            f"| {short(ds)} | {ds_type(ds)} "
            f"| {pct(b.get('precision'))} | {pct(b.get('recall'))} | {pct(b.get('f1'))} "
            f"| {pct(l.get('precision'))} | {pct(l.get('recall'))} | {pct(l.get('f1'))} "
            f"| {delta*100:+.1f}% |"
        )

    ai, li = avg_f1(base, int_ds), avg_f1(lora, int_ds)
    ae, le = avg_f1(base, ext_ds), avg_f1(lora, ext_ds)

    lines += [
        "",
        "## Summary",
        "",
        "| Split | Base Avg F1 | LoRA Avg F1 | Gain |",
        "|-------|------------:|------------:|-----:|",
        f"| Internal (4 datasets) | {ai*100:.1f}% | {li*100:.1f}% | {(li-ai)*100:+.1f}% |",
        f"| External (6 datasets) | {ae*100:.1f}% | {le*100:.1f}% | {(le-ae)*100:+.1f}% |",
        "",
        "## Notes",
        "",
        "- **External datasets** (ADE Corpus v2, PHEE, DDI-2013, TAC-2017 ADR, CADEC) "
        "were never seen during training.",
        "- **DDI-2013**: LoRA recall improves substantially but precision drops, "
        "indicating the model over-predicts relations on this dataset.",
        "- **TAC-2017 / CADEC**: harder domain transfer; LoRA shows improved recall over the base model.",
        "- The base model occasionally fails to produce valid JSON output "
        "(parse_success_rate < 1.0 on some datasets).",
    ]
    return "\n".join(lines)


# -- Main entry point ---------------------------------------------------------

def main() -> None:
    """Load benchmark results and write comparison reports."""
    base = load_csv(BASE_CSV)
    lora = load_csv(LORA_CSV)

    txt = build_txt(base, lora)
    md  = build_md(base, lora)

    print(txt)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "quantitative_report.txt").write_text(txt, encoding="utf-8")
    (OUTPUT_DIR / "quantitative_report.md").write_text(md, encoding="utf-8")

    print(f"\nSaved -> ../MISC/results/quantitative_report.txt")
    print(f"Saved -> ../MISC/results/quantitative_report.md")


if __name__ == "__main__":
    main()
