from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import TrainerCallback


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def collect_runtime_environment() -> Dict[str, Any]:
    environment: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
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
    def __init__(self, output_path: Path):
        self.output_path = output_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "logs": logs,
        }
        append_jsonl(self.output_path, payload)
