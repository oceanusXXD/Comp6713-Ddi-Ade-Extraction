"""推理后端适配层。

这个模块把仓库的推理逻辑抽象成两种后端：
- `transformers`：标准 Hugging Face 逐批生成
- `vllm`：高吞吐推理路径

无论走哪种后端，最终都会产出统一结构的 prediction row。
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

import torch
from transformers import AutoModelForCausalLM

from src.model_utils import (
    enforce_hf_offline_mode,
    load_tokenizer,
    suppress_local_flash_attn_shadowing,
    torch_dtype_from_name,
)
from src.parser import (
    DatasetExample,
    canonicalize_prediction_row,
    parse_prediction_text,
)
from src.prompting import apply_chat_template, build_messages

LOGGER = logging.getLogger(__name__)
DEFAULT_API_KEY_ENV_NAMES = ("QWEN_API_KEY", "DASHSCOPE_API_KEY", "ALIYUN_API_KEY")


def resolve_tokenizer_source(model_config: Dict[str, Any]) -> str:
    """决定 tokenizer 应该优先从 adapter 还是 base model 路径加载。"""
    tokenizer_source = model_config.get("tokenizer_name_or_path")
    if tokenizer_source is not None:
        return str(tokenizer_source)
    return str(model_config["base_model_name_or_path"])


def load_model_and_tokenizer_transformers(config: Dict[str, Any]):
    """加载 transformers 推理后端所需的 model 与 tokenizer。"""
    model_config = config["model"]
    adapter_path = model_config.get("adapter_path")
    tokenizer_source = resolve_tokenizer_source(model_config)
    try:
        tokenizer = load_tokenizer(
            {
                "model_name_or_path": tokenizer_source,
                "trust_remote_code": model_config.get("trust_remote_code", True),
            }
        )
    except Exception:
        if adapter_path is None or tokenizer_source == model_config["base_model_name_or_path"]:
            raise
        LOGGER.info("Tokenizer not found in adapter path; falling back to base model tokenizer.")
        tokenizer = load_tokenizer(
            {
                "model_name_or_path": model_config["base_model_name_or_path"],
                "trust_remote_code": model_config.get("trust_remote_code", True),
            }
        )
    tokenizer.padding_side = "left"

    requested_dtype = torch_dtype_from_name(model_config.get("torch_dtype"))
    if not torch.cuda.is_available() and requested_dtype in {torch.float16, torch.bfloat16}:
        LOGGER.info("CUDA unavailable; overriding torch dtype %s -> float32 for inference.", requested_dtype)
        requested_dtype = torch.float32

    model_source = str(model_config["base_model_name_or_path"])
    local_files_only = Path(model_source).expanduser().exists()
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": model_config.get("trust_remote_code", True),
        "torch_dtype": requested_dtype,
        "local_files_only": local_files_only,
    }
    if model_config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_config["attn_implementation"]
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    with suppress_local_flash_attn_shadowing(
        enabled=str(model_config.get("attn_implementation", "")).lower() == "flash_attention_2"
    ):
        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "Adapter loading requested but peft is not installed. Install peft or remove model.adapter_path."
            ) from exc
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)

    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))
    model.eval()
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = True
    return model, tokenizer


def ensure_local_vllm_import(vllm_module: Any) -> None:
    """确保导入到的是当前虚拟环境里的 vLLM，而不是系统残留版本。"""
    module_path = Path(vllm_module.__file__).resolve()
    venv_prefix = Path(sys.prefix).resolve()
    if not str(module_path).startswith(str(venv_prefix)):
        raise RuntimeError(
            f"vLLM must be imported from the active virtualenv. "
            f"Resolved {module_path}, expected under {venv_prefix}."
        )


@contextmanager
def suppress_broken_flash_attn_detection() -> Any:
    """在导入 vLLM 时临时屏蔽本地 `flash_attn` 兼容层的探测。"""
    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str, package: Optional[str] = None):
        if name == "flash_attn" or name.startswith("flash_attn."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = patched_find_spec
    try:
        yield
    finally:
        importlib.util.find_spec = original_find_spec


def load_model_and_tokenizer_vllm(config: Dict[str, Any]):
    """加载 vLLM 推理后端。"""
    model_config = config["model"]
    adapter_path = model_config.get("adapter_path")
    tokenizer_source = resolve_tokenizer_source(model_config)
    enforce_hf_offline_mode(allow_remote=bool(config.get("allow_remote_model_source", False)))

    tokenizer = load_tokenizer(
        {
            "model_name_or_path": tokenizer_source,
            "trust_remote_code": model_config.get("trust_remote_code", True),
            "allow_remote_model_source": config.get("allow_remote_model_source", False),
        }
    )
    tokenizer.padding_side = "left"

    with suppress_broken_flash_attn_detection():
        import vllm as vllm_module
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

    ensure_local_vllm_import(vllm_module)

    vllm_config = config.get("vllm", {})
    max_model_len = vllm_config.get("max_model_len")
    if max_model_len is None:
        max_model_len = config["inference"]["max_input_length"] + config["inference"]["max_new_tokens"]

    llm = LLM(
        model=str(model_config["base_model_name_or_path"]),
        tokenizer=tokenizer_source,
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        dtype=str(model_config.get("torch_dtype", "bfloat16")),
        tensor_parallel_size=int(vllm_config.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(vllm_config.get("gpu_memory_utilization", 0.85)),
        max_model_len=int(max_model_len),
        enable_lora=adapter_path is not None,
        max_lora_rank=int(vllm_config.get("max_lora_rank", 64)),
        max_loras=int(vllm_config.get("max_loras", 1)),
        enforce_eager=bool(vllm_config.get("enforce_eager", False)),
        disable_log_stats=bool(vllm_config.get("disable_log_stats", True)),
        hf_overrides={"local_files_only": not bool(config.get("allow_remote_model_source", False))},
        hf_token=False,
    )
    return llm, tokenizer, SamplingParams, LoRARequest


def load_runtime_dotenv(project_root: Path) -> None:
    """Load available `.env` files when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    candidates = [
        project_root / ".env",
        project_root / "Echo" / "CODE" / ".env",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=True)


def resolve_api_key(config: Dict[str, Any]) -> str:
    """Resolve API key from config or environment."""
    project_root = Path(config.get("config_path", Path.cwd())).resolve().parent.parent
    load_runtime_dotenv(project_root)

    api_config = config.get("api", {})
    inline_api_key = str(api_config.get("api_key") or "").strip()
    if inline_api_key:
        return inline_api_key

    candidate_env_names: List[str] = []
    configured_env_name = str(api_config.get("api_key_env") or "").strip()
    if configured_env_name:
        candidate_env_names.append(configured_env_name)
    for env_name in DEFAULT_API_KEY_ENV_NAMES:
        if env_name not in candidate_env_names:
            candidate_env_names.append(env_name)

    for env_name in candidate_env_names:
        value = os.getenv(env_name)
        if value and value.strip():
            return value.strip()

    raise RuntimeError("API key not configured.")


def resolve_api_model_name(config: Dict[str, Any]) -> str:
    """Resolve the effective remote model id for the request."""
    api_config = config.get("api", {})
    adapter_path = config["model"].get("adapter_path")
    if adapter_path is None:
        return str(api_config.get("base_model_name") or "").strip()
    return str(api_config.get("lora_model_name") or "").strip()


def load_model_and_tokenizer_api(config: Dict[str, Any]):
    """Prepare an OpenAI-compatible API bundle."""
    api_config = config.get("api", {})
    bundle = {
        "base_url": str(api_config.get("base_url") or "").rstrip("/"),
        "timeout_seconds": int(api_config.get("timeout_seconds", 120)),
    }
    return bundle, None


def extract_api_response_text(payload: Dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-compatible chat completion payload."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()
    raise ValueError("API response did not include choices[0].message.content.")


def call_openai_compatible_chat_api(
    *,
    config: Dict[str, Any],
    api_bundle: Dict[str, Any],
    example: DatasetExample,
) -> str:
    """Call the configured OpenAI-compatible chat completion endpoint."""
    inference_config = config["inference"]
    model_name = resolve_api_model_name(config)
    if not model_name:
        raise RuntimeError("No remote model id resolved for backend=api.")

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": build_messages(example.system_prompt, example.user_text),
        "max_tokens": int(inference_config["max_new_tokens"]),
        "stream": False,
    }
    if bool(inference_config.get("do_sample", False)):
        payload["temperature"] = float(inference_config.get("temperature", 0.0))
        payload["top_p"] = float(inference_config.get("top_p", 1.0))
    else:
        payload["temperature"] = 0.0

    request = urllib_request.Request(
        url=f"{api_bundle['base_url']}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {resolve_api_key(config)}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=api_bundle["timeout_seconds"]) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {error_body[:800]}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc.reason}") from exc

    try:
        parsed_body = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API response was not valid JSON: {body[:800]}") from exc

    return extract_api_response_text(parsed_body)


def model_input_device(model: Any) -> torch.device:
    """返回 transformers 模型当前主输入设备。"""
    return next(model.parameters()).device


def build_prompt_text(example: DatasetExample, tokenizer: Any, enable_thinking: Optional[bool]) -> str:
    """把一条样本组装成最终送入生成模型的 prompt 文本。"""
    return apply_chat_template(
        tokenizer,
        build_messages(example.system_prompt, example.user_text),
        tokenize=False,
        add_generation_prompt=True,
        truncation=False,
        enable_thinking=enable_thinking,
    )


def build_prediction_row(
    *,
    example: DatasetExample,
    raw_output: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """把原始生成文本包装成统一 prediction row。"""
    parsed = parse_prediction_text(raw_output)
    adapter_path = config["model"].get("adapter_path")
    return canonicalize_prediction_row(
        {
            "sample_id": example.sample_id,
            "split": example.split,
            "text": example.user_text,
            "system_prompt": example.system_prompt,
            "gold_relations": example.gold_relations,
            "raw_output": raw_output,
            "raw_json_candidate": parsed.raw_candidate,
            "predicted_relations": parsed.relations,
            "parse_status": parsed.status,
            "parse_failure_reason": parsed.failure_reason,
            "model_name_or_path": config["model"]["base_model_name_or_path"],
            "adapter_path": str(adapter_path) if adapter_path is not None else None,
        }
    )


def generate_predictions_transformers(
    model: Any,
    tokenizer: Any,
    examples: Sequence[DatasetExample],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """使用 Hugging Face `generate` 跑批量推理。"""
    inference_config = config["inference"]
    enable_thinking = config["model"].get("enable_thinking")
    device = model_input_device(model)
    results: List[Dict[str, Any]] = []

    for start in range(0, len(examples), inference_config["batch_size"]):
        batch_examples = list(examples[start : start + inference_config["batch_size"]])
        prompt_texts = [build_prompt_text(example, tokenizer, enable_thinking) for example in batch_examples]
        encoded = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=inference_config["max_input_length"],
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        prompt_lengths = encoded["attention_mask"].sum(dim=1)

        # 推理参数尽量与配置文件一一对应，便于在日志和实验结果里回溯。
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": inference_config["max_new_tokens"],
            "do_sample": bool(inference_config.get("do_sample", False)),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": inference_config.get("repetition_penalty", 1.0),
        }
        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = inference_config["temperature"]
            generation_kwargs["top_p"] = inference_config["top_p"]

        with torch.inference_mode():
            generated = model.generate(**encoded, **generation_kwargs)

        for index, example in enumerate(batch_examples):
            output_tokens = generated[index][int(prompt_lengths[index]) :]
            raw_output = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            results.append(build_prediction_row(example=example, raw_output=raw_output, config=config))

        LOGGER.info("Processed %s/%s samples.", len(results), len(examples))

    return results


def generate_predictions_vllm(
    llm: Any,
    tokenizer: Any,
    examples: Sequence[DatasetExample],
    config: Dict[str, Any],
    sampling_params_class: Any,
    lora_request_class: Any,
) -> List[Dict[str, Any]]:
    """使用 vLLM 跑批量推理。"""
    inference_config = config["inference"]
    enable_thinking = config["model"].get("enable_thinking")
    adapter_path = config["model"].get("adapter_path")
    base_model_name_or_path = str(config["model"]["base_model_name_or_path"])
    results: List[Dict[str, Any]] = []

    sampling_kwargs: Dict[str, Any] = {
        "temperature": float(inference_config["temperature"]) if inference_config.get("do_sample", False) else 0.0,
        "top_p": float(inference_config.get("top_p", 1.0)),
        "max_tokens": int(inference_config["max_new_tokens"]),
        "repetition_penalty": float(inference_config.get("repetition_penalty", 1.0)),
    }
    if tokenizer.eos_token_id is not None:
        sampling_kwargs["stop_token_ids"] = [int(tokenizer.eos_token_id)]
    sampling_params = sampling_params_class(**sampling_kwargs)

    lora_request = None
    if adapter_path is not None:
        lora_request = lora_request_class(
            "final_adapter",
            1,
            str(adapter_path),
            base_model_name=base_model_name_or_path,
        )

    for start in range(0, len(examples), inference_config["batch_size"]):
        batch_examples = list(examples[start : start + inference_config["batch_size"]])
        prompt_texts = [build_prompt_text(example, tokenizer, enable_thinking) for example in batch_examples]
        outputs = llm.generate(
            prompt_texts,
            sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )

        for example, output in zip(batch_examples, outputs):
            output_text = ""
            if output.outputs:
                output_text = output.outputs[0].text.strip()
            results.append(build_prediction_row(example=example, raw_output=output_text, config=config))

        LOGGER.info("Processed %s/%s samples.", len(results), len(examples))

    return results


def generate_predictions_api(
    api_bundle: Dict[str, Any],
    _tokenizer: Any,
    examples: Sequence[DatasetExample],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run prediction requests through an OpenAI-compatible API backend."""
    results: List[Dict[str, Any]] = []
    selected_model_name = resolve_api_model_name(config)

    for index, example in enumerate(examples, start=1):
        raw_output = call_openai_compatible_chat_api(
            config=config,
            api_bundle=api_bundle,
            example=example,
        )
        row = build_prediction_row(example=example, raw_output=raw_output, config=config)
        row["model_name_or_path"] = selected_model_name
        results.append(row)
        LOGGER.info("Processed %s/%s samples via api backend.", index, len(examples))

    return results


def generate_predictions(
    model_bundle: Any,
    tokenizer: Any,
    examples: Sequence[DatasetExample],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """根据配置自动分发到对应推理后端。"""
    backend = str(config.get("backend", "transformers")).lower()
    if backend == "vllm":
        llm, sampling_params_class, lora_request_class = model_bundle
        return generate_predictions_vllm(
            llm,
            tokenizer,
            examples,
            config,
            sampling_params_class=sampling_params_class,
            lora_request_class=lora_request_class,
        )
    if backend == "api":
        return generate_predictions_api(model_bundle, tokenizer, examples, config)
    return generate_predictions_transformers(model_bundle, tokenizer, examples, config)
