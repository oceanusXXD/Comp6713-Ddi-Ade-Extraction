from __future__ import annotations

"""Evaluate prediction JSONL files against packaged ADE/DDI gold labels."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import read_jsonl
from src.parser import (
    canonicalize_prediction_row,
    evaluate_prediction_rows,
    format_metrics_report,
    load_dataset_examples,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate ADE/DDI prediction jsonl files.")
    parser.add_argument(
        "--predictions-path",
        type=str,
        required=True,
        help="Prediction jsonl path produced by scripts/inference/predict.py or legacy pretest scripts.",
    )
    parser.add_argument(
        "--gold-path",
        type=str,
        default=None,
        help="Optional dataset jsonl path to attach gold labels when predictions do not include them.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        help="Split label used when loading --gold-path for sample ids.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional text report output path. Defaults to <predictions>_metrics.txt.",
    )
    parser.add_argument(
        "--json-output-path",
        type=str,
        default=None,
        help="Optional JSON metrics output path. Defaults to <predictions>_metrics.json.",
    )
    return parser.parse_args()


def default_output_path(predictions_path: Path, suffix: str) -> Path:
    """Build a default output filename from the prediction file path."""
    return predictions_path.with_name(f"{predictions_path.stem}{suffix}")


def attach_gold_labels(
    prediction_rows: List[Dict[str, Any]],
    *,
    gold_path: Optional[Path],
    split: str,
) -> List[Dict[str, Any]]:
    """Attach gold labels when the prediction file does not already contain them."""
    if gold_path is None:
        return prediction_rows

    gold_examples = load_dataset_examples(gold_path, split=split)
    gold_by_sample_id = {example.sample_id: example.gold_relations for example in gold_examples}

    for index, row in enumerate(prediction_rows):
        if row.get("gold_relations") is not None:
            continue
        sample_id = row.get("sample_id")
        if sample_id in gold_by_sample_id:
            row["gold_relations"] = gold_by_sample_id[sample_id]
        elif index < len(gold_examples):
            row["gold_relations"] = gold_examples[index].gold_relations
            row.setdefault("sample_id", gold_examples[index].sample_id)
        else:
            row["gold_relations"] = []

    return prediction_rows


def main() -> None:
    """Run the evaluation workflow."""
    args = parse_args()
    predictions_path = Path(args.predictions_path).expanduser().resolve()
    gold_path = Path(args.gold_path).expanduser().resolve() if args.gold_path else None

    raw_rows = read_jsonl(predictions_path)
    raw_rows = attach_gold_labels(raw_rows, gold_path=gold_path, split=args.split)
    normalized_rows = [canonicalize_prediction_row(row) for row in raw_rows]

    metrics = evaluate_prediction_rows(normalized_rows)
    report_text = format_metrics_report(metrics, prediction_path=predictions_path)

    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else default_output_path(predictions_path, "_metrics.txt")
    )
    json_output_path = (
        Path(args.json_output_path).expanduser().resolve()
        if args.json_output_path
        else default_output_path(predictions_path, "_metrics.json")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report_text)
    print(f"Saved text report to: {output_path}")
    print(f"Saved JSON metrics to: {json_output_path}")


if __name__ == "__main__":
    main()
