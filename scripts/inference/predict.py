"""仓库主推理入口。

支持两类用法：
1. 读取 JSONL 数据集做批量推理并顺手算指标。
2. 直接输入一段原始文本做单样本推理。
"""

from __future__ import annotations

import argparse
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
    return parser.parse_args()


def configure_logging() -> None:
    """初始化日志。"""
    logging.basicConfig(
        level=logging.INFO,
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


def main() -> None:
    """推理脚本主流程。"""
    configure_logging()
    args = parse_args()
    config = apply_cli_overrides(load_inference_config(args.config, validate=False), args)
    if args.enable_thinking and args.disable_thinking:
        raise ValueError("Use at most one of --enable-thinking or --disable-thinking.")
    if args.enable_thinking:
        config["model"]["enable_thinking"] = True
    elif args.disable_thinking:
        config["model"]["enable_thinking"] = False
    backend = str(config.get("backend", "transformers")).lower()
    set_global_seed(config["seed"], seed_cuda=backend != "vllm")

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

    LOGGER.info("Loaded %s examples.", len(examples))
    LOGGER.info("Using inference backend: %s", backend)
    if backend == "vllm":
        llm, tokenizer, sampling_params_class, lora_request_class = load_model_and_tokenizer_vllm(config)
        model_bundle = (llm, sampling_params_class, lora_request_class)
    else:
        model_bundle, tokenizer = load_model_and_tokenizer_transformers(config)
    rows = generate_predictions(model_bundle, tokenizer, examples, config)

    predictions_path = config["output"].get("predictions_path")
    if predictions_path is not None:
        write_jsonl(predictions_path, rows)
        LOGGER.info("Saved predictions to %s", predictions_path)

    metrics = write_metrics_if_available(rows, config)
    if metrics is not None:
        LOGGER.info("Micro F1: %.4f", metrics["micro"]["f1"])


if __name__ == "__main__":
    main()
