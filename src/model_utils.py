from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "resume_from_checkpoint": None,
    "model_name_or_path": "../models/Qwen3-8B",
    "allow_remote_model_source": False,
    "trust_remote_code": True,
    "torch_dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "enable_thinking": False,
    "system_prompt_path": None,
    "train_path": "data/merged_chatml_train.jsonl",
    "validation_path": "data/merged_chatml_validation.jsonl",
    "max_seq_length": 4096,
    "finetune_type": "lora",
    "load_in_4bit": False,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "modules_to_save": None,
    "output_dir": "outputs/qwen3_8b_lora",
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 3,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 2,
    "gradient_checkpointing": True,
    "remove_unused_columns": False,
    "report_to": [],
    "bf16": True,
    "fp16": False,
    "max_grad_norm": 1.0,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}


def deep_merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_project_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def try_resolve_existing_path(value: str) -> Optional[Path]:
    raw_path = Path(value).expanduser()
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((PROJECT_ROOT / raw_path).resolve())
        candidates.append(raw_path.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_model_source(value: Optional[str], *, allow_remote: bool = False) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    existing_path = try_resolve_existing_path(normalized)
    if existing_path is not None:
        return str(existing_path)
    if allow_remote:
        return normalized
    raise FileNotFoundError(
        "Remote model sources are disabled for this repository. "
        f"Provide a local model path that exists on disk: {normalized}"
    )


def validate_local_model_source(value: Optional[str], label: str, *, allow_remote: bool = False) -> None:
    if value is None:
        return
    if allow_remote:
        return
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.exists():
        raise FileNotFoundError(f"{label} must be a local path that exists on disk: {value}")


def torch_dtype_from_name(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name in (None, "", "auto"):
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def normalize_report_to(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null"}:
            return []
        return [value]
    if isinstance(value, list):
        return value
    raise ValueError(f"Unsupported report_to value: {value!r}")


def flatten_legacy_sections(loaded: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(loaded)

    model = config.pop("model", None)
    if isinstance(model, dict):
        config.setdefault("model_name_or_path", model.get("name_or_path"))
        config.setdefault("trust_remote_code", model.get("trust_remote_code"))
        config.setdefault("torch_dtype", model.get("torch_dtype"))
        config.setdefault("attn_implementation", model.get("attn_implementation"))
        config.setdefault("enable_thinking", model.get("enable_thinking"))

    data = config.pop("data", None)
    if isinstance(data, dict):
        config.setdefault("train_path", data.get("train_path"))
        config.setdefault("validation_path", data.get("validation_path"))
        config.setdefault("max_seq_length", data.get("max_seq_length"))
        config.setdefault("system_prompt_path", data.get("system_prompt_path"))

    training = config.pop("training", None)
    if isinstance(training, dict):
        for key, value in training.items():
            config.setdefault(key, value)

    peft = config.pop("peft", None)
    if isinstance(peft, dict):
        if "finetune_type" not in config:
            if not peft.get("enabled", True):
                config["finetune_type"] = "full"
            elif peft.get("use_qlora", False):
                config["finetune_type"] = "qlora"
            else:
                config["finetune_type"] = "lora"
        config.setdefault("load_in_4bit", peft.get("use_qlora"))
        config.setdefault("lora_r", peft.get("lora_r"))
        config.setdefault("lora_alpha", peft.get("lora_alpha"))
        config.setdefault("lora_dropout", peft.get("lora_dropout"))
        config.setdefault("target_modules", peft.get("target_modules"))
        config.setdefault("bias", peft.get("bias"))
        config.setdefault("task_type", peft.get("task_type"))
        config.setdefault("modules_to_save", peft.get("modules_to_save"))
        config.setdefault("bnb_4bit_quant_type", peft.get("bnb_4bit_quant_type"))
        config.setdefault("bnb_4bit_compute_dtype", peft.get("bnb_4bit_compute_dtype"))
        config.setdefault("bnb_4bit_use_double_quant", peft.get("bnb_4bit_use_double_quant"))

    return config


def load_training_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Training config must be a YAML mapping.")

    config = deep_merge_dict(DEFAULT_CONFIG, flatten_legacy_sections(loaded))
    config["config_path"] = config_path
    config["model_name_or_path"] = resolve_model_source(
        config.get("model_name_or_path"),
        allow_remote=bool(config.get("allow_remote_model_source", False)),
    )
    config["train_path"] = resolve_project_path(config["train_path"])
    config["validation_path"] = resolve_project_path(config.get("validation_path"))
    config["system_prompt_path"] = resolve_project_path(config.get("system_prompt_path"))
    config["output_dir"] = resolve_project_path(config["output_dir"])
    config["resume_from_checkpoint"] = resolve_project_path(config.get("resume_from_checkpoint"))
    config["report_to"] = normalize_report_to(config.get("report_to"))
    validate_training_config(config)
    return config


def validate_training_config(config: Dict[str, Any]) -> None:
    validate_local_model_source(
        config.get("model_name_or_path"),
        "Model path",
        allow_remote=bool(config.get("allow_remote_model_source", False)),
    )

    train_path = config["train_path"]
    if train_path is None or not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    validation_path = config.get("validation_path")
    if validation_path is not None and not validation_path.exists():
        raise FileNotFoundError(f"Validation data not found: {validation_path}")

    system_prompt_path = config.get("system_prompt_path")
    if system_prompt_path is not None and not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt path not found: {system_prompt_path}")

    if config["max_seq_length"] <= 0:
        raise ValueError("max_seq_length must be a positive integer.")

    output_dir = config["output_dir"]
    if output_dir is None:
        raise ValueError("output_dir is required.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    finetune_type = str(config["finetune_type"]).lower()
    if finetune_type not in {"full", "lora", "qlora"}:
        raise ValueError("finetune_type must be one of: full, lora, qlora.")
    config["finetune_type"] = finetune_type

    if finetune_type == "qlora" and not config["load_in_4bit"]:
        raise ValueError("finetune_type=qlora requires load_in_4bit=true.")
    if finetune_type != "qlora" and config["load_in_4bit"]:
        raise ValueError("load_in_4bit=true is only supported with finetune_type=qlora.")
    if finetune_type in {"lora", "qlora"} and not config["target_modules"]:
        raise ValueError("LoRA finetuning requires non-empty target_modules.")


def enforce_hf_offline_mode(*, allow_remote: bool) -> None:
    if allow_remote:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def load_tokenizer(config: Dict[str, Any]):
    model_name_or_path = config.get("model_name_or_path") or config.get("name_or_path")
    if not model_name_or_path:
        raise KeyError("model_name_or_path")
    enforce_hf_offline_mode(allow_remote=bool(config.get("allow_remote_model_source", False)))
    local_files_only = Path(model_name_or_path).expanduser().exists()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=config.get("trust_remote_code", True),
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_quantization_config(config: Dict[str, Any]):
    if not config.get("load_in_4bit", False):
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("QLoRA requires transformers BitsAndBytesConfig support.") from exc

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError("QLoRA requested but bitsandbytes is not installed.") from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch_dtype_from_name(config["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
    )


def load_model(config: Dict[str, Any]):
    enforce_hf_offline_mode(allow_remote=bool(config.get("allow_remote_model_source", False)))
    quantization_config = build_quantization_config(config)
    model_name_or_path = config["model_name_or_path"]
    local_files_only = Path(model_name_or_path).expanduser().exists()

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": config.get("trust_remote_code", True),
        "torch_dtype": torch_dtype_from_name(config.get("torch_dtype")),
        "local_files_only": local_files_only,
    }
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config["attn_implementation"]
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    return model


def apply_peft_if_requested(model: Any, config: Dict[str, Any]) -> Any:
    if config["finetune_type"] == "full":
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError(
            "LoRA/QLoRA requested but peft is not installed. Install peft to train adapters."
        ) from exc

    if config["finetune_type"] == "qlora":
        model = prepare_model_for_kbit_training(model)

    task_type = getattr(TaskType, config.get("task_type", "CAUSAL_LM"))
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"],
        task_type=task_type,
        target_modules=config["target_modules"],
        modules_to_save=config.get("modules_to_save"),
    )
    model = get_peft_model(model, lora_config)

    if config.get("gradient_checkpointing", False) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model


def build_model_and_tokenizer(config: Dict[str, Any], tokenizer: Any = None):
    if tokenizer is None:
        tokenizer = load_tokenizer(config)
    model = load_model(config)
    model = apply_peft_if_requested(model, config)
    return model, tokenizer


def save_adapter_artifacts(model: Any, tokenizer: Any, output_dir: Path) -> Path:
    adapter_dir = output_dir / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir
