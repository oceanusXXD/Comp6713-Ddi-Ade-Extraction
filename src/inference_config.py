"""Inference-config loading, validation, and CLI override helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.model_utils import (
    PROJECT_ROOT,
    deep_merge_dict,
    resolve_model_source,
    resolve_project_path,
    validate_local_model_source,
)

DEFAULT_INFER_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "system_prompt_path": None,
    "backend": "vllm",
    "allow_remote_model_source": False,
    "model": {
        "base_model_name_or_path": "../models/Qwen3-8B",
        "tokenizer_name_or_path": None,
        "adapter_path": None,
        "trust_remote_code": True,
        "torch_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "enable_thinking": False,
    },
    "data": {
        "split": "dev",
        "input_path": "data/merged_chatml_validation.jsonl",
        "max_samples": None,
    },
    "inference": {
        "batch_size": 8,
        "max_input_length": 4096,
        "max_new_tokens": 2048,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    },
    "vllm": {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 6144,
        "max_lora_rank": 64,
        "max_loras": 1,
        "enforce_eager": False,
        "disable_log_stats": True,
    },
    "api": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "QWEN_API_KEY",
        "api_key": None,
        "timeout_seconds": 120,
        "base_model_name": "qwen2.5-14b-instruct",
        "lora_model_name": "qwen3.6-max-preview",
    },
    "output": {
        "predictions_path": "results/inference_runs/qwen3_8b_dev_predictions.jsonl",
        "metrics_path": "results/inference_runs/qwen3_8b_dev_metrics.txt",
        "metrics_json_path": "results/inference_runs/qwen3_8b_dev_metrics.json",
    },
}


def split_to_default_path(split: str) -> Path:
    """Map a short split alias to the package's default dataset path."""
    normalized = split.strip().lower()
    if normalized == "dev":
        normalized = "validation"
    if normalized not in {"validation", "test"}:
        raise ValueError("Only dev/validation/test dataset splits are supported for batch inference.")
    return (PROJECT_ROOT / "data" / f"merged_chatml_{normalized}.jsonl").resolve()


def resolve_output_path(value: Optional[str]) -> Optional[Path]:
    """Resolve an output file path relative to the project root."""
    return resolve_project_path(value)


def load_inference_config(config_path: str | Path, *, validate: bool = True) -> Dict[str, Any]:
    """Load the inference config and resolve runtime paths."""
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Inference config must be a YAML mapping.")

    config = deep_merge_dict(DEFAULT_INFER_CONFIG, loaded)
    config["config_path"] = config_path
    if config["data"].get("split") is not None:
        config["data"]["split"] = str(config["data"]["split"])
    config["system_prompt_path"] = resolve_project_path(config.get("system_prompt_path"))

    backend = str(config.get("backend", "transformers")).lower()
    allow_remote = bool(config.get("allow_remote_model_source", False))

    if backend == "api":
        if config["model"].get("base_model_name_or_path") is not None:
            config["model"]["base_model_name_or_path"] = str(config["model"]["base_model_name_or_path"]).strip()
        if config["model"].get("tokenizer_name_or_path") is not None:
            config["model"]["tokenizer_name_or_path"] = str(config["model"]["tokenizer_name_or_path"]).strip()
        if config["model"].get("adapter_path") is not None:
            config["model"]["adapter_path"] = str(config["model"]["adapter_path"]).strip() or None
        config["api"]["base_url"] = str(config["api"].get("base_url") or "").rstrip("/")
        if config["api"].get("api_key_env") is not None:
            config["api"]["api_key_env"] = str(config["api"]["api_key_env"]).strip()
        if config["api"].get("api_key") is not None:
            config["api"]["api_key"] = str(config["api"]["api_key"]).strip() or None
        config["api"]["base_model_name"] = str(config["api"].get("base_model_name") or "").strip()
        config["api"]["lora_model_name"] = str(config["api"].get("lora_model_name") or "").strip()
    else:
        config["model"]["base_model_name_or_path"] = resolve_model_source(
            config["model"].get("base_model_name_or_path"),
            allow_remote=allow_remote,
        )
        config["model"]["tokenizer_name_or_path"] = resolve_model_source(
            config["model"].get("tokenizer_name_or_path"),
            allow_remote=allow_remote,
        )
        config["model"]["adapter_path"] = resolve_project_path(config["model"].get("adapter_path"))

    config["data"]["input_path"] = resolve_project_path(config["data"].get("input_path"))
    config["output"]["predictions_path"] = resolve_output_path(config["output"].get("predictions_path"))
    config["output"]["metrics_path"] = resolve_output_path(config["output"].get("metrics_path"))
    config["output"]["metrics_json_path"] = resolve_output_path(config["output"].get("metrics_json_path"))
    if validate:
        validate_inference_config(config)
    return config


def validate_inference_config(config: Dict[str, Any]) -> None:
    """Validate the critical inference settings before execution starts."""
    backend = str(config.get("backend", "transformers")).lower()
    if backend not in {"transformers", "vllm", "api"}:
        raise ValueError("backend must be one of: transformers, vllm, api.")

    if backend == "api":
        api_config = config.get("api", {})
        if not str(api_config.get("base_url") or "").strip():
            raise ValueError("api.base_url must be a non-empty URL for backend=api.")
        if not str(api_config.get("base_model_name") or "").strip():
            raise ValueError("api.base_model_name must be a non-empty model id for backend=api.")
        if not str(api_config.get("lora_model_name") or "").strip():
            raise ValueError("api.lora_model_name must be a non-empty model id for backend=api.")
        if int(api_config.get("timeout_seconds", 0)) <= 0:
            raise ValueError("api.timeout_seconds must be a positive integer.")
    else:
        allow_remote = bool(config.get("allow_remote_model_source", False))
        validate_local_model_source(
            config["model"].get("base_model_name_or_path"),
            "Base model path",
            allow_remote=allow_remote,
        )
        validate_local_model_source(
            config["model"].get("tokenizer_name_or_path"),
            "Tokenizer path",
            allow_remote=allow_remote,
        )
        adapter_path = config["model"].get("adapter_path")
        if adapter_path is not None and not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    if config["inference"]["batch_size"] <= 0:
        raise ValueError("inference.batch_size must be a positive integer.")
    if config["inference"]["max_input_length"] <= 0:
        raise ValueError("inference.max_input_length must be a positive integer.")
    if config["inference"]["max_new_tokens"] <= 0:
        raise ValueError("inference.max_new_tokens must be a positive integer.")
    if config["data"].get("input_path") is not None and not config["data"]["input_path"].exists():
        raise FileNotFoundError(f"Input dataset not found: {config['data']['input_path']}")
    if config.get("system_prompt_path") is not None and not config["system_prompt_path"].exists():
        raise FileNotFoundError(f"System prompt path not found: {config['system_prompt_path']}")


def apply_cli_overrides(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Apply CLI overrides on top of a loaded inference config."""
    merged = deep_merge_dict(config, {})
    if args.backend is not None:
        merged["backend"] = args.backend
    if args.split is not None:
        merged["data"]["split"] = args.split
        if args.input_path is None:
            merged["data"]["input_path"] = split_to_default_path(args.split)
    if args.input_path is not None:
        merged["data"]["input_path"] = resolve_project_path(args.input_path)
    if args.system_prompt_path is not None:
        merged["system_prompt_path"] = resolve_project_path(args.system_prompt_path)
    if args.limit is not None:
        merged["data"]["max_samples"] = args.limit
    if args.batch_size is not None:
        merged["inference"]["batch_size"] = args.batch_size
    if args.max_new_tokens is not None:
        merged["inference"]["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        merged["inference"]["temperature"] = args.temperature
        merged["inference"]["do_sample"] = args.temperature > 0.0

    backend = str(merged.get("backend", "transformers")).lower()
    if args.base_model is not None:
        if backend == "api":
            merged["model"]["base_model_name_or_path"] = str(args.base_model).strip()
            merged["api"]["base_model_name"] = str(args.base_model).strip()
        else:
            merged["model"]["base_model_name_or_path"] = resolve_model_source(
                args.base_model,
                allow_remote=bool(merged.get("allow_remote_model_source", False)),
            )
    if args.adapter_path is not None:
        if backend == "api":
            merged["model"]["adapter_path"] = str(args.adapter_path).strip() or None
            if merged["model"]["adapter_path"] is not None:
                merged["api"]["lora_model_name"] = str(args.adapter_path).strip()
        else:
            merged["model"]["adapter_path"] = resolve_project_path(args.adapter_path)
    if args.output_path is not None:
        merged["output"]["predictions_path"] = resolve_output_path(args.output_path)
    if args.metrics_path is not None:
        merged["output"]["metrics_path"] = resolve_output_path(args.metrics_path)
    if args.metrics_json_path is not None:
        merged["output"]["metrics_json_path"] = resolve_output_path(args.metrics_json_path)
    validate_inference_config(merged)
    return merged
