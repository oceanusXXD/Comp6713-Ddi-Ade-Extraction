"""Main batch and single-text inference entrypoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.inference_backends import (
    generate_predictions,
    load_model_and_tokenizer_transformers,
    load_model_and_tokenizer_vllm,
)
from src.inference_config import apply_cli_overrides, load_inference_config
from src.parser import (
    DatasetExample,
    evaluate_prediction_rows,
    format_metrics_report,
    load_dataset_examples,
)
from src.prompting import load_system_prompt

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run ADE/DDI inference with the packaged Echo configs.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml",
        help="Path to the inference YAML config.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("transformers", "vllm"),
        default=None,
        help="Optional backend override.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional dataset split alias override (validation/test).",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Optional dataset jsonl path override.",
    )
    parser.add_argument(
        "--input-text",
        type=str,
        default=None,
        help="Run one-off inference on a single medical text instead of a dataset file.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Inline system prompt override. Intended for use with --input-text.",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=str,
        default=None,
        help="Optional system prompt file override.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of dataset rows to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional inference batch size override.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional max_new_tokens override.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature override. Any value > 0 enables sampling.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override the base model path expected by the config.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Override the LoRA adapter path.",
    )
    parser.add_argument(
        "--disable-adapter",
        action="store_true",
        help="Ignore the configured adapter and run the base model only.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional prediction jsonl output path override.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Optional text metrics output path override.",
    )
    parser.add_argument(
        "--metrics-json-path",
        type=str,
        default=None,
        help="Optional JSON metrics output path override.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(*, debug: bool) -> None:
    """Configure the entrypoint logger."""

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_global_seed(seed: int, *, seed_cuda: bool = True) -> None:
    """Set Python and Torch RNG seeds."""

    random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_single_example(text: str, *, system_prompt: str) -> DatasetExample:
    """Build an in-memory single-text example."""

    return DatasetExample(
        sample_id="single_0000",
        split="single",
        system_prompt=(system_prompt or "").strip(),
        user_text=text.strip(),
        gold_relations=None,
    )


def _preview_text(text: str, *, limit: int = 240) -> str:
    """Return a truncated single-line preview."""

    compact = " ".join((text or "").strip().split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def build_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load the YAML config and apply CLI overrides."""

    config = load_inference_config(args.config)
    config = apply_cli_overrides(config, args)

    if args.disable_adapter:
        config["model"]["adapter_path"] = None

    if args.input_text is not None:
        config["data"]["split"] = "single"
        config["data"]["input_path"] = None
        config["data"]["max_samples"] = 1
        if args.output_path is None:
            config["output"]["predictions_path"] = (
                PROJECT_ROOT.parent / "MISC" / "results" / "inference_runs" / "single_text_predictions.jsonl"
            ).resolve()
        config["output"]["metrics_path"] = None
        config["output"]["metrics_json_path"] = None

    return config


def resolve_single_prompt(config: Dict[str, Any], args: argparse.Namespace) -> str:
    """Resolve the system prompt for single-text inference."""

    if args.system_prompt is not None:
        return args.system_prompt.strip()

    prompt_path = None
    if config.get("system_prompt_path") is not None:
        prompt_path = str(config["system_prompt_path"])
    return load_system_prompt(prompt_path)


def build_examples(config: Dict[str, Any], args: argparse.Namespace) -> List[DatasetExample]:
    """Build either a dataset-backed example list or a single in-memory example."""

    if args.input_text is not None:
        system_prompt = resolve_single_prompt(config, args)
        return [build_single_example(args.input_text, system_prompt=system_prompt)]

    input_path = config["data"].get("input_path")
    if input_path is None:
        raise ValueError("Dataset inference requires a resolved input_path in the config or via --input-path.")

    return load_dataset_examples(
        Path(input_path),
        split=str(config["data"].get("split") or "dataset"),
        limit=config["data"].get("max_samples"),
    )


def load_generation_bundle(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Load the selected inference backend and tokenizer."""

    backend = str(config.get("backend", "transformers")).lower()
    if backend == "vllm":
        llm, tokenizer, sampling_params_class, lora_request_class = load_model_and_tokenizer_vllm(config)
        return (llm, sampling_params_class, lora_request_class), tokenizer

    model, tokenizer = load_model_and_tokenizer_transformers(config)
    return model, tokenizer


def log_runtime_configuration(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Log the resolved runtime configuration."""

    model_config = config["model"]
    inference_config = config["inference"]
    output_config = config["output"]
    LOGGER.info("Loaded config from %s", config.get("config_path"))
    LOGGER.info(
        "Resolved model sources: backend=%s base_model=%s tokenizer=%s adapter=%s",
        config.get("backend"),
        model_config.get("base_model_name_or_path"),
        model_config.get("tokenizer_name_or_path"),
        model_config.get("adapter_path"),
    )
    if args.input_text is not None:
        LOGGER.info(
            "Input mode: single_text chars=%s prompt_override=%s preview=%s",
            len(args.input_text),
            bool(args.system_prompt or args.system_prompt_path),
            _preview_text(args.input_text, limit=320),
        )
    else:
        LOGGER.info(
            "Input mode: dataset split=%s path=%s limit=%s",
            config["data"].get("split"),
            config["data"].get("input_path"),
            config["data"].get("max_samples"),
        )
    LOGGER.info(
        "Generation config: batch_size=%s max_input_length=%s max_new_tokens=%s do_sample=%s temperature=%s top_p=%s repetition_penalty=%s",
        inference_config.get("batch_size"),
        inference_config.get("max_input_length"),
        inference_config.get("max_new_tokens"),
        inference_config.get("do_sample"),
        inference_config.get("temperature"),
        inference_config.get("top_p"),
        inference_config.get("repetition_penalty"),
    )
    LOGGER.info(
        "Output paths: predictions=%s metrics=%s metrics_json=%s",
        output_config.get("predictions_path"),
        output_config.get("metrics_path"),
        output_config.get("metrics_json_path"),
    )
    LOGGER.debug(
        "Runtime cwd=%s cuda_available=%s cuda_device_count=%s",
        Path.cwd(),
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write normalized prediction rows to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def maybe_write_metrics(config: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Write aggregate metrics when the rows contain gold labels."""

    if not rows or not any(row.get("gold_relations") is not None for row in rows):
        LOGGER.info("No gold relations attached; skipping metrics output.")
        return None

    metrics = evaluate_prediction_rows(rows)
    prediction_path = config["output"].get("predictions_path")
    metrics_path = config["output"].get("metrics_path")
    metrics_json_path = config["output"].get("metrics_json_path")

    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            format_metrics_report(metrics, prediction_path=prediction_path),
            encoding="utf-8",
        )
        LOGGER.info("Saved metrics report to %s", metrics_path)

    if metrics_json_path is not None:
        metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("Saved metrics JSON to %s", metrics_json_path)

    return metrics


def summarize_prediction_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a compact runtime summary over prediction rows."""

    total_samples = len(rows)
    parsed_samples = sum(1 for row in rows if row.get("parse_status") == "parsed")
    nonempty_samples = sum(1 for row in rows if row.get("predicted_relations"))
    predicted_relation_total = sum(len(row.get("predicted_relations") or []) for row in rows)
    failure_counts = Counter(
        row.get("parse_failure_reason") or "unknown"
        for row in rows
        if row.get("parse_status") != "parsed"
    )
    return {
        "total_samples": total_samples,
        "parsed_samples": parsed_samples,
        "parse_success_rate": (parsed_samples / total_samples) if total_samples else 0.0,
        "nonempty_samples": nonempty_samples,
        "predicted_nonempty_rate": (nonempty_samples / total_samples) if total_samples else 0.0,
        "mean_predicted_relations": (predicted_relation_total / total_samples) if total_samples else 0.0,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def log_prediction_summary(rows: Sequence[Dict[str, Any]]) -> None:
    """Log a compact summary of inference results."""

    summary = summarize_prediction_rows(rows)
    LOGGER.info(
        "Prediction summary: total=%s parsed=%s parse_success_rate=%.3f nonempty=%s predicted_nonempty_rate=%.3f mean_predicted_relations=%.3f failure_counts=%s",
        summary["total_samples"],
        summary["parsed_samples"],
        summary["parse_success_rate"],
        summary["nonempty_samples"],
        summary["predicted_nonempty_rate"],
        summary["mean_predicted_relations"],
        summary["failure_counts"],
    )

    if len(rows) == 1:
        row = rows[0]
        LOGGER.info(
            "Single-sample result: sample_id=%s parse_status=%s failure_reason=%s predicted_relations=%s",
            row.get("sample_id"),
            row.get("parse_status"),
            row.get("parse_failure_reason"),
            json.dumps(row.get("predicted_relations") or [], ensure_ascii=False),
        )
        LOGGER.info("Single-sample raw output preview: %s", _preview_text(str(row.get("raw_output", "")), limit=600))
        return

    failure_rows = [row for row in rows if row.get("parse_status") != "parsed"][:3]
    for row in failure_rows:
        LOGGER.warning(
            "Parse failure sample: sample_id=%s reason=%s raw_output_preview=%s",
            row.get("sample_id"),
            row.get("parse_failure_reason"),
            _preview_text(str(row.get("raw_output", "")), limit=400),
        )


def main() -> None:
    """Run the packaged inference workflow."""

    args = parse_args()
    configure_logging(debug=args.debug)

    try:
        config = build_runtime_config(args)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n"
            "Download the base model first with:\n"
            "bash scripts/setup/download_base_model.sh"
        ) from exc
    set_global_seed(int(config.get("seed", 42)))
    log_runtime_configuration(config, args)

    examples = build_examples(config, args)
    if not examples:
        raise ValueError("No inference examples were loaded.")

    model_bundle, tokenizer = load_generation_bundle(config)
    rows = generate_predictions(model_bundle, tokenizer, examples, config)

    prediction_path = config["output"].get("predictions_path")
    if prediction_path is not None:
        write_jsonl(prediction_path, rows)
        LOGGER.info("Saved predictions to %s", prediction_path)

    metrics = maybe_write_metrics(config, rows)
    log_prediction_summary(rows)

    if len(rows) == 1:
        print(json.dumps(rows[0].get("predicted_relations") or [], ensure_ascii=False, indent=2))
        return

    if prediction_path is not None:
        print(f"Saved predictions to: {prediction_path}")
    if metrics is not None:
        print(format_metrics_report(metrics, prediction_path=prediction_path))


if __name__ == "__main__":
    main()
