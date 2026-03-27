from __future__ import annotations

"""训练与推理可观测性工具。

这里的职责比较单一：把环境信息、参数统计和训练日志稳定写到磁盘，
方便后续复盘“当时到底是在什么环境、什么配置下跑出来的”。
"""

import importlib
import json
from pathlib import Path
import site
import sys
from typing import Any, Dict

import torch
from transformers import TrainerCallback


def json_default(value: Any) -> Any:
    """为 Path / torch.dtype 这类对象提供 JSON 序列化兼容。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    """写格式化 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """向 JSONL 文件追加一条记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def collect_runtime_environment() -> Dict[str, Any]:
    """收集当前 Python、CUDA 和核心依赖包来源。"""
    environment: Dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "usersite": site.getusersitepackages(),
            "user_site_enabled": site.ENABLE_USER_SITE,
        },
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    package_origins: Dict[str, Any] = {}
    for package_name in ("torch", "transformers", "peft", "vllm"):
        try:
            module = importlib.import_module(package_name)
            package_origins[package_name] = getattr(module, "__file__", None)
        except Exception as exc:  # pragma: no cover - observability only
            package_origins[package_name] = {"import_error": repr(exc)}
    environment["package_origins"] = package_origins
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        environment["devices"] = devices
    return environment


def collect_parameter_statistics(model: Any) -> Dict[str, Any]:
    """统计模型总参数量、可训练参数量和可训练比例。"""
    total_parameters = 0
    trainable_parameters = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total_parameters += count
        if parameter.requires_grad:
            trainable_parameters += count
    trainable_ratio = trainable_parameters / total_parameters if total_parameters else 0.0
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_ratio": trainable_ratio,
    }


class JsonlMetricsCallback(TrainerCallback):
    """把 Trainer 的日志事件同步落成 JSONL。"""
    def __init__(self, output_path: Path):
        self.output_path = output_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次 Trainer 触发 log 事件时追加一条记录。"""
        if not logs:
            return
        payload = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "logs": logs,
        }
        append_jsonl(self.output_path, payload)
