"""Inference backend adapters.

This module exposes the repository inference flow through two backends:
`transformers` for standard Hugging Face generation and `vllm` for higher
throughput inference. Both produce the same prediction-row structure.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def resolve_tokenizer_source(model_config: Dict[str, Any]) -> str:
    """Choose whether tokenizer loading should prefer the adapter or base model path."""
    tokenizer_source = model_config.get("tokenizer_name_or_path")
    if tokenizer_source is not None:
        return str(tokenizer_source)
    return str(model_config["base_model_name_or_path"])


def load_model_and_tokenizer_transformers(config: Dict[str, Any]):
    """Load the model and tokenizer needed for the transformers backend."""
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
    """Ensure the imported vLLM module comes from the active virtual environment."""
    module_path = Path(vllm_module.__file__).resolve()
    venv_prefix = Path(sys.prefix).resolve()
    if not str(module_path).startswith(str(venv_prefix)):
        raise RuntimeError(
            f"vLLM must be imported from the active virtualenv. "
            f"Resolved {module_path}, expected under {venv_prefix}."
        )


@contextmanager
def suppress_broken_flash_attn_detection() -> Any:
    """Temporarily suppress local `flash_attn` detection while importing vLLM."""
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
    """Load the vLLM inference backend."""
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


def model_input_device(model: Any) -> torch.device:
    """Return the primary input device for a transformers model."""
    return next(model.parameters()).device


def build_prompt_text(example: DatasetExample, tokenizer: Any, enable_thinking: Optional[bool]) -> str:
    """Assemble one example into the final prompt text for generation."""
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
    """Wrap raw generated text into a normalized prediction row."""
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
    """Run batched inference with Hugging Face `generate`."""
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

        # Keep generation arguments aligned with config values for easier tracing.
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
    """Run batched inference with vLLM."""
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


def generate_predictions(
    model_bundle: Any,
    tokenizer: Any,
    examples: Sequence[DatasetExample],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Dispatch inference to the configured backend."""
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
    return generate_predictions_transformers(model_bundle, tokenizer, examples, config)
