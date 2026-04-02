"""仓库主推理入口。

支持两类用法：
1. 读取 JSONL 数据集做批量推理并顺手算指标。
2. 直接输入一段原始文本做单样本推理。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.inference_backends import (
    generate_predictions,
    load_model_and_tokenizer_transformers,
    load_model_and_tokenizer_vllm,
)
from src.inference_config import (
    apply_cli_overrides,
    load_inference_config,
    split_to_default_path,
)
from src.parser import (
    DatasetExample,
    evaluate_prediction_rows,
    format_metrics_report,
    load_dataset_examples,
)
from src.prompting import load_system_prompt

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析推理命令行参数。"""
    parser = argparse.ArgumentParser(description="Run ADE/DDI extraction inference and evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/infer_qwen3_8b_lora_ddi_ade_final.yaml",
        help="Path to the inference YAML config.",
    )
    parser.add_argument("--split", type=str, default=None, help="Dataset split alias: dev|validation|test.")
    parser.add_argument("--input-path", type=str, default=None, help="Override input dataset path.")
    parser.add_argument("--input-text", type=str, default=None, help="Run single-sample inference on raw text.")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Inference backend: transformers or vllm.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt override for single-sample inference.",
    )
    parser.add_argument("--output-path", type=str, default=None, help="Override prediction jsonl output path.")
    parser.add_argument("--metrics-path", type=str, default=None, help="Override text metrics output path.")
    parser.add_argument(
        "--system-prompt-path",
        type=str,
        default=None,
        help="Optional file path overriding the default structured extraction system prompt.",
    )
    parser.add_argument(
        "--metrics-json-path",
        type=str,
        default=None,
        help="Override JSON metrics output path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of dataset samples.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation length.")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature.")
    parser.add_argument("--base-model", type=str, default=None, help="Override base model name or path.")
    parser.add_argument("--adapter-path", type=str, default=None, help="Override LoRA adapter path.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Override config to enable tokenizer chat-template thinking mode.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Override config to disable tokenizer chat-template thinking mode.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging for config resolution, inference stages, and parse summaries.",
    )
    return parser.parse_args()


def configure_logging(*, debug: bool = False) -> None:
    """初始化日志。"""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_global_seed(seed: int, *, seed_cuda: bool = True) -> None:
    """设置推理阶段随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_single_example(
    text: str,
    *,
    system_prompt: Optional[str],
) -> DatasetExample:
    """把单条原始文本包装成与批量推理一致的样本对象。"""
    return DatasetExample(
        sample_id="single_0000",
        split="single",
        system_prompt=(system_prompt or "").strip(),
        user_text=text.strip(),
        gold_relations=[],
    )


def _preview_text(text: str, *, limit: int = 240) -> str:
    """把多行文本压平成适合日志输出的短摘要。"""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def log_effective_config(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """输出解析后的关键推理配置，便于定位路径和开关问题。"""
    model_config = config["model"]
    inference_config = config["inference"]
    output_config = config["output"]
    LOGGER.info("Loaded config from %s", config.get("config_path"))
    LOGGER.info(
        "Resolved model sources: backend=%s allow_remote=%s base_model=%s tokenizer=%s adapter=%s",
        config.get("backend"),
        config.get("allow_remote_model_source"),
        model_config.get("base_model_name_or_path"),
        model_config.get("tokenizer_name_or_path"),
        model_config.get("adapter_path"),
    )
    if args.input_text is not None:
        LOGGER.info(
            "Input mode: single_text chars=%s system_prompt_override=%s",
            len(args.input_text),
            bool(args.system_prompt),
        )
        LOGGER.debug("Single input preview: %s", _preview_text(args.input_text, limit=500))
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
    LOGGER.debug("Runtime cwd=%s cuda_available=%s cuda_device_count=%s", Path.cwd(), torch.cuda.is_available(), torch.cuda.device_count())


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """把预测结果写成 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics_if_available(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """如果预测结果里带 gold，就顺手计算并写出指标。"""
    if not rows:
        return None
    if not all("gold_relations" in row for row in rows):
        LOGGER.info("No gold relations found in predictions; skipping metrics output.")
        return None

    metrics = evaluate_prediction_rows(rows)
    metrics_path = config["output"].get("metrics_path")
    metrics_json_path = config["output"].get("metrics_json_path")
    prediction_path = config["output"].get("predictions_path")

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
    """汇总解析状态和预测非空率，便于快速判断推理异常。"""
    total_rows = len(rows)
    parsed_rows = 0
    nonempty_rows = 0
    relation_total = 0
    failure_counts: Counter[str] = Counter()

    for row in rows:
        parse_status = str(row.get("parse_status", "parsed"))
        if parse_status == "parsed":
            parsed_rows += 1
        else:
            failure_reason = row.get("parse_failure_reason")
            failure_counts[str(failure_reason) if failure_reason else "unknown"] += 1
        predicted_relations = row.get("predicted_relations") or []
        if predicted_relations:
            nonempty_rows += 1
        relation_total += len(predicted_relations)

    return {
        "total_samples": total_rows,
        "parsed_samples": parsed_rows,
        "parse_success_rate": (parsed_rows / total_rows) if total_rows else 0.0,
        "nonempty_samples": nonempty_rows,
        "predicted_nonempty_rate": (nonempty_rows / total_rows) if total_rows else 0.0,
        "mean_predicted_relations": (relation_total / total_rows) if total_rows else 0.0,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def log_prediction_debug_summary(rows: Sequence[Dict[str, Any]]) -> None:
    """输出推理结果摘要，并对单样本/解析失败样本给出更具体的日志。"""
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
    if not rows:
        return

    if len(rows) == 1:
        row = rows[0]
        predicted_relations = row.get("predicted_relations") or []
        LOGGER.info(
            "Single-sample result: sample_id=%s parse_status=%s failure_reason=%s predicted_relations=%s",
            row.get("sample_id"),
            row.get("parse_status"),
            row.get("parse_failure_reason"),
            json.dumps(predicted_relations, ensure_ascii=False),
        )
        LOGGER.info("Single-sample raw output preview: %s", _preview_text(str(row.get("raw_output", "")), limit=800))
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
    """推理脚本主流程。"""
    args = parse_args()
    configure_logging(debug=args.debug)
    try:
        config = apply_cli_overrides(load_inference_config(args.config, validate=False), args)
    except Exception:
        LOGGER.exception("Failed to load/resolve inference config from %s", args.config)
        raise
    if args.enable_thinking and args.disable_thinking:
        raise ValueError("Use at most one of --enable-thinking or --disable-thinking.")
    if args.enable_thinking:
        config["model"]["enable_thinking"] = True
    elif args.disable_thinking:
        config["model"]["enable_thinking"] = False
    log_effective_config(config, args)
    backend = str(config.get("backend", "transformers")).lower()
    set_global_seed(config["seed"], seed_cuda=backend != "vllm")

    try:
        if args.input_text is not None:
            # 单文本模式：不依赖数据文件，直接构造一个临时样本。
            system_prompt = args.system_prompt or load_system_prompt(
                str(config["system_prompt_path"]) if config.get("system_prompt_path") is not None else None
            )
            examples = [
                build_single_example(
                    args.input_text,
                    system_prompt=system_prompt,
                )
            ]
        else:
            # 数据集模式：加载文件中的 gold_relations，便于推理后立即评估。
            input_path = config["data"].get("input_path")
            if input_path is None:
                split = config["data"].get("split", "dev")
                input_path = split_to_default_path(split)
                config["data"]["input_path"] = input_path
            split_name = str(config["data"].get("split") or input_path.stem).lower()
            examples = load_dataset_examples(
                input_path,
                split=split_name,
                limit=config["data"].get("max_samples"),
            )
            system_prompt = load_system_prompt(
                str(config["system_prompt_path"]) if config.get("system_prompt_path") is not None else None
            )
            examples = [
                DatasetExample(
                    sample_id=example.sample_id,
                    split=example.split,
                    system_prompt=system_prompt,
                    user_text=example.user_text,
                    gold_relations=example.gold_relations,
                )
                for example in examples
            ]
    except Exception:
        LOGGER.exception(
            "Failed while preparing inference examples: input_text_mode=%s input_path=%s split=%s",
            args.input_text is not None,
            config["data"].get("input_path"),
            config["data"].get("split"),
        )
        raise

    LOGGER.info("Loaded %s examples.", len(examples))
    LOGGER.info("Using inference backend: %s", backend)
    LOGGER.info("Initializing model runtime...")
    try:
        if backend == "vllm":
            llm, tokenizer, sampling_params_class, lora_request_class = load_model_and_tokenizer_vllm(config)
            model_bundle = (llm, sampling_params_class, lora_request_class)
        else:
            model_bundle, tokenizer = load_model_and_tokenizer_transformers(config)
    except Exception:
        LOGGER.exception(
            "Failed to initialize inference runtime: backend=%s base_model=%s adapter=%s",
            backend,
            config["model"].get("base_model_name_or_path"),
            config["model"].get("adapter_path"),
        )
        raise

    try:
        rows = generate_predictions(model_bundle, tokenizer, examples, config)
    except Exception:
        LOGGER.exception(
            "Inference generation failed: backend=%s examples=%s batch_size=%s",
            backend,
            len(examples),
            config["inference"].get("batch_size"),
        )
        raise
    log_prediction_debug_summary(rows)

    predictions_path = config["output"].get("predictions_path")
    if predictions_path is not None:
        try:
            write_jsonl(predictions_path, rows)
        except Exception:
            LOGGER.exception("Failed to write predictions to %s", predictions_path)
            raise
        LOGGER.info("Saved predictions to %s", predictions_path)

    try:
        metrics = write_metrics_if_available(rows, config)
    except Exception:
        LOGGER.exception(
            "Failed to compute/write metrics: metrics_path=%s metrics_json_path=%s",
            config["output"].get("metrics_path"),
            config["output"].get("metrics_json_path"),
        )
        raise
    if metrics is not None:
        LOGGER.info("Micro F1: %.4f", metrics["micro"]["f1"])


if __name__ == "__main__":
    main()
