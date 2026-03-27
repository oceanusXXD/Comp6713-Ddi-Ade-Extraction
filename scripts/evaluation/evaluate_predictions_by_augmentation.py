"""按增强类型拆分评估的辅助脚本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import read_jsonl
from src.parser import canonicalize_prediction_row, evaluate_prediction_rows, format_metrics_report


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate predictions grouped by augmentation_type.")
    parser.add_argument("--predictions-path", type=str, required=True, help="Prediction JSONL path.")
    parser.add_argument(
        "--source-path",
        type=str,
        required=True,
        help="Source dataset JSONL path containing augmentation_type for each row.",
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        default=None,
        help="Optional grouped metrics JSON output path.",
    )
    parser.add_argument(
        "--output-md-path",
        type=str,
        default=None,
        help="Optional grouped metrics markdown output path.",
    )
    return parser.parse_args()


def default_output_path(path: Path, suffix: str) -> Path:
    """根据输入文件名生成默认输出文件名。"""
    return path.with_name(f"{path.stem}{suffix}")


def main() -> None:
    """按 augmentation_type 分组计算预测指标。"""
    args = parse_args()
    predictions_path = Path(args.predictions_path).expanduser().resolve()
    source_path = Path(args.source_path).expanduser().resolve()

    prediction_rows = [canonicalize_prediction_row(row) for row in read_jsonl(predictions_path)]
    source_rows = read_jsonl(source_path)

    if len(prediction_rows) != len(source_rows):
        raise ValueError(
            f"Predictions/source length mismatch: {len(prediction_rows)} vs {len(source_rows)}"
        )

    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    for prediction_row, source_row in zip(prediction_rows, source_rows):
        aug_type = source_row.get("augmentation_type")
        if aug_type is None:
            raise ValueError(f"source row missing augmentation_type: {source_row}")
        grouped_rows.setdefault(str(aug_type), []).append(prediction_row)

    grouped_metrics = {
        aug_type: evaluate_prediction_rows(rows)
        for aug_type, rows in sorted(grouped_rows.items())
    }

    output_json_path = (
        Path(args.output_json_path).expanduser().resolve()
        if args.output_json_path
        else default_output_path(predictions_path, "_by_augmentation.json")
    )
    output_md_path = (
        Path(args.output_md_path).expanduser().resolve()
        if args.output_md_path
        else default_output_path(predictions_path, "_by_augmentation.md")
    )

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(grouped_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    sections = ["# Grouped Augmentation Metrics"]
    for aug_type, metrics in grouped_metrics.items():
        sections.append(f"\n## {aug_type}\n")
        sections.append(format_metrics_report(metrics, prediction_path=predictions_path))
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text("\n".join(sections), encoding="utf-8")

    print(json.dumps(grouped_metrics, ensure_ascii=False, indent=2))
    print(f"Saved grouped JSON to: {output_json_path}")
    print(f"Saved grouped markdown to: {output_md_path}")


if __name__ == "__main__":
    main()
