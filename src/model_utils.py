"""训练配置、模型加载和 PEFT 包装工具。

这个模块主要解决三件事：
1. 把 YAML 配置和历史配置格式统一整理成当前训练配置。
2. 统一处理本地模型路径、远程模型开关、量化参数和 tokenizer 加载。
3. 在需要时给基础模型套上 LoRA / QLoRA 适配器。
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import sys
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
    "use_rslora": False,
    "use_dora": False,
    "init_lora_weights": True,
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
    "label_smoothing_factor": 0.0,
    "train_sampling_strategy": "none",
    "empty_target_sampling_weight": 1.0,
    "ddi_sampling_weight": 1.0,
    "ddi_int_sampling_weight": 1.0,
    "multi_relation_sampling_weight": 1.0,
    "use_sample_loss_weights": False,
    "empty_target_loss_weight": 1.0,
    "ddi_loss_weight": 1.0,
    "ddi_int_loss_weight": 1.0,
    "multi_relation_loss_weight": 1.0,
    "adapter_optimizer": "default",
    "loraplus_lr_ratio": 16.0,
    "loraplus_lr_embedding": 1e-6,
    "loraplus_weight_decay": None,
}


def deep_merge_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置字典，保留 base 中未被覆盖的默认值。"""
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_project_path(value: Optional[str]) -> Optional[Path]:
    """把相对仓库路径解析成绝对路径。"""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def try_resolve_existing_path(value: str) -> Optional[Path]:
    """尽量把用户给的字符串解析成当前机器上真实存在的路径。"""
    raw_path = Path(value).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((PROJECT_ROOT / raw_path).resolve())
        candidates.append(raw_path.resolve())
        # 文档约定是「仓库上一级 /models/模型名」，但实际常见把模型放在「仓库根 /models/模型名」
        candidates.append((PROJECT_ROOT / "models" / raw_path.name).resolve())
        # 若写的是 models/Qwen3-8B（相对仓库根）
        if raw_path.parts and raw_path.parts[0] == "models" and len(raw_path.parts) > 1:
            candidates.append((PROJECT_ROOT.joinpath(*raw_path.parts)).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_hf_id_if_local_qwen3_missing(normalized: str) -> Optional[str]:
    """当配置写的是默认本地 Qwen3-8B 布局但磁盘上未找到时，返回官方 HF id 供 allow_remote 兜底。"""
    lower = normalized.replace("\\", "/").rstrip("/")
    if lower.endswith("Qwen3-8B") or lower.endswith("models/Qwen3-8B"):
        return "Qwen/Qwen3-8B"
    return None


def resolve_model_source(value: Optional[str], *, allow_remote: bool = False) -> Optional[str]:
    """解析模型来源。

    默认仓库只允许本地路径；只有显式打开 `allow_remote_model_source`
    时才允许保留 Hugging Face model id 这类远程来源。
    """
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    existing_path = try_resolve_existing_path(normalized)
    if existing_path is not None:
        return str(existing_path)
    if allow_remote:
        fallback = _default_hf_id_if_local_qwen3_missing(normalized)
        if fallback is not None:
            return fallback
        return normalized
    raise FileNotFoundError(
        "Remote model sources are disabled for this repository. "
        f"Provide a local model path that exists on disk: {normalized}"
    )


def validate_local_model_source(value: Optional[str], label: str, *, allow_remote: bool = False) -> None:
    """校验模型来源是否合法。"""
    if value is None:
        return
    if allow_remote:
        return
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.exists():
        raise FileNotFoundError(f"{label} must be a local path that exists on disk: {value}")


def torch_dtype_from_name(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    """把配置里的字符串 dtype 转成 torch.dtype。"""
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
    """把 report_to 规范成 transformers 可接受的列表格式。"""
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


def normalize_init_lora_weights(value: Any) -> Any:
    """兼容布尔值和字符串形式的 LoRA 初始化配置。"""
    if value is None:
        return True
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return lowered
    return value


def flatten_legacy_sections(loaded: Dict[str, Any]) -> Dict[str, Any]:
    """把旧版分节配置展平成当前统一字段。

    这个仓库经历过几轮配置重构，所以这里负责兼容：
    - model/data/training 分节
    - 旧版 peft 小节
    - 早期的 qlora / lora 开关
    """
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
        config.setdefault("use_rslora", peft.get("use_rslora"))
        config.setdefault("use_dora", peft.get("use_dora"))
        config.setdefault("init_lora_weights", peft.get("init_lora_weights"))
        config.setdefault("target_modules", peft.get("target_modules"))
        config.setdefault("bias", peft.get("bias"))
        config.setdefault("task_type", peft.get("task_type"))
        config.setdefault("modules_to_save", peft.get("modules_to_save"))
        config.setdefault("adapter_optimizer", peft.get("adapter_optimizer"))
        config.setdefault("loraplus_lr_ratio", peft.get("loraplus_lr_ratio"))
        config.setdefault("loraplus_lr_embedding", peft.get("loraplus_lr_embedding"))
        config.setdefault("loraplus_weight_decay", peft.get("loraplus_weight_decay"))
        config.setdefault("bnb_4bit_quant_type", peft.get("bnb_4bit_quant_type"))
        config.setdefault("bnb_4bit_compute_dtype", peft.get("bnb_4bit_compute_dtype"))
        config.setdefault("bnb_4bit_use_double_quant", peft.get("bnb_4bit_use_double_quant"))

    return config


def load_training_config(config_path: str | Path) -> Dict[str, Any]:
    """读取 YAML 训练配置，补默认值、做路径解析并完成校验。"""
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
    """对训练配置做集中校验，尽量在训练前提前失败。"""
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

    adapter_optimizer = str(config.get("adapter_optimizer", "default")).strip().lower()
    if adapter_optimizer in {"", "none", "null"}:
        adapter_optimizer = "default"
    if adapter_optimizer not in {"default", "loraplus", "lorafa"}:
        raise ValueError("adapter_optimizer must be one of: default, loraplus, lorafa.")
    config["adapter_optimizer"] = adapter_optimizer

    # 下面这组约束的目的，是避免“配置看起来合法，但组合起来不成立”的情况。
    if finetune_type == "qlora" and not config["load_in_4bit"]:
        raise ValueError("finetune_type=qlora requires load_in_4bit=true.")
    if finetune_type != "qlora" and config["load_in_4bit"]:
        raise ValueError("load_in_4bit=true is only supported with finetune_type=qlora.")
    if finetune_type in {"lora", "qlora"} and not config["target_modules"]:
        raise ValueError("LoRA finetuning requires non-empty target_modules.")
    if finetune_type == "full" and adapter_optimizer != "default":
        raise ValueError("adapter_optimizer is only supported when finetune_type uses LoRA adapters.")

    config["use_rslora"] = bool(config.get("use_rslora", False))
    config["use_dora"] = bool(config.get("use_dora", False))
    config["init_lora_weights"] = normalize_init_lora_weights(config.get("init_lora_weights", True))
    init_lora_weights = config["init_lora_weights"]
    if isinstance(init_lora_weights, str):
        if init_lora_weights == "eva":
            raise ValueError(
                "init_lora_weights=eva is not wired into this repository. EVA requires an extra activation "
                "initialization pass before training."
            )
        if init_lora_weights == "corda" or init_lora_weights.startswith("corda"):
            raise ValueError(
                "init_lora_weights=corda is not wired into this repository. CorDA requires preprocessing before "
                "training."
            )
        if init_lora_weights == "loftq":
            raise ValueError(
                "init_lora_weights=loftq is not wired into this repository. LoftQ requires a separate quantized "
                "initialization flow."
            )

    sampling_strategy = str(config.get("train_sampling_strategy", "none")).lower()
    if sampling_strategy not in {"none", "weighted"}:
        raise ValueError("train_sampling_strategy must be one of: none, weighted.")
    config["train_sampling_strategy"] = sampling_strategy

    if float(config.get("loraplus_lr_ratio", 16.0)) < 1.0:
        raise ValueError("loraplus_lr_ratio must be >= 1.0.")
    if float(config.get("loraplus_lr_embedding", 1e-6)) < 0.0:
        raise ValueError("loraplus_lr_embedding must be >= 0.")
    loraplus_weight_decay = config.get("loraplus_weight_decay")
    if loraplus_weight_decay is not None and float(loraplus_weight_decay) < 0.0:
        raise ValueError("loraplus_weight_decay must be >= 0 when set.")

    if float(config.get("label_smoothing_factor", 0.0)) < 0.0:
        raise ValueError("label_smoothing_factor must be >= 0.")

    for key in (
        "empty_target_sampling_weight",
        "ddi_sampling_weight",
        "ddi_int_sampling_weight",
        "multi_relation_sampling_weight",
        "empty_target_loss_weight",
        "ddi_loss_weight",
        "ddi_int_loss_weight",
        "multi_relation_loss_weight",
    ):
        if float(config.get(key, 1.0)) <= 0.0:
            raise ValueError(f"{key} must be > 0.")


def enforce_hf_offline_mode(*, allow_remote: bool) -> None:
    """在禁远程模式下，强制 Hugging Face 走离线路径解析。"""
    if allow_remote:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


@contextmanager
def suppress_local_flash_attn_shadowing(*, enabled: bool):
    """临时从 `sys.path` 中拿掉仓库根目录，避免本地兼容层误伤真实依赖。

    只有在需要 `flash_attention_2` 这类真实第三方实现时才启用。
    """
    if not enabled:
        yield
        return

    original_sys_path = list(sys.path)
    filtered_sys_path = []
    for entry in original_sys_path:
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:
            resolved = None
        if resolved == PROJECT_ROOT:
            continue
        filtered_sys_path.append(entry)

    sys.path[:] = filtered_sys_path
    try:
        yield
    finally:
        sys.path[:] = original_sys_path


def load_tokenizer(config: Dict[str, Any]):
    """按训练配置加载 tokenizer，并补齐 pad token。"""
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
    """在 QLoRA 模式下构造 bitsandbytes 量化配置。"""
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
    """加载基础 Causal LM 模型，并按配置应用 dtype / 量化 / attention 设置。"""
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

    # 某些环境里仓库内的本地 `flash_attn` 兼容层会遮蔽真实三方包，
    # 这里在需要时临时拿掉这种遮蔽，确保 HF 能加载真实实现。
    with suppress_local_flash_attn_shadowing(
        enabled=str(config.get("attn_implementation", "")).lower() == "flash_attention_2"
    ):
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    return model


def apply_peft_if_requested(model: Any, config: Dict[str, Any]) -> Any:
    """根据配置决定是否给模型套上 LoRA / QLoRA。"""
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
        use_rslora=bool(config.get("use_rslora", False)),
        use_dora=bool(config.get("use_dora", False)),
        init_lora_weights=config.get("init_lora_weights", True),
    )
    model = get_peft_model(model, lora_config)

    if config.get("gradient_checkpointing", False) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model


def build_model_and_tokenizer(config: Dict[str, Any], tokenizer: Any = None):
    """统一构建训练所需的 model 与 tokenizer。"""
    if tokenizer is None:
        tokenizer = load_tokenizer(config)
    model = load_model(config)
    model = apply_peft_if_requested(model, config)
    return model, tokenizer


def save_adapter_artifacts(model: Any, tokenizer: Any, output_dir: Path) -> Path:
    """把最终 adapter 和 tokenizer 资产写到 `final_adapter/`。"""
    adapter_dir = output_dir / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir
