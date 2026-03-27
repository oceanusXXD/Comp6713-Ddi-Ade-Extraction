"""LoRA 变体批量实验脚本。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_utils import load_training_config
import yaml

VARIANT_CONFIGS: Dict[str, str] = {
    "final": "configs/qwen3_8b_lora_ddi_ade_final.yaml",
}

AUGMENT_DIR = "data/processed/Comp6713-Ddi-Ade-Extraction_final_augment"
DEFAULT_PROMPT = "prompts/medical_relation_extraction_system_prompt.txt"
DEFAULT_INFER_CONFIG = "configs/infer_qwen3_8b_lora_ddi_ade_final.yaml"


def parse_args() -> argparse.Namespace:
    """解析实验脚本命令行参数。"""
    parser = argparse.ArgumentParser(description="Train and evaluate Qwen3-8B LoRA variant experiments.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANT_CONFIGS),
        choices=list(VARIANT_CONFIGS),
        help="Variants to run.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=("transformers", "vllm"),
        help="Inference backend used for validation/test prediction.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=None,
        help="Optional runtime override for num_train_epochs during benchmark training.",
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only run inference/evaluation.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation inference/evaluation.")
    parser.add_argument("--skip-test", action="store_true", help="Skip test inference/evaluation.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for smoke training runs.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional cap for smoke eval during train.")
    parser.add_argument("--prediction-limit", type=int, default=None, help="Optional cap for validation/test prediction.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/variant_benchmark_qwen3_8b",
        help="Directory for benchmark outputs and logs.",
    )
    return parser.parse_args()


def run_command(command: List[str], log_path: Path) -> None:
    """执行外部命令并把日志写到文件。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(command)} | log={log_path}")


def materialize_runtime_config(config: Dict[str, Any], path: Path) -> Path:
    """把运行时配置落盘，保证实验可复现。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(config)
    for key, value in list(serializable.items()):
        if isinstance(value, Path):
            serializable[key] = str(value)
    path.write_text(yaml.safe_dump(serializable, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def metric_cell(metrics: Dict[str, Any], key: str) -> float:
    """从统一指标结构中提取指定指标值。"""
    if key == "micro_f1":
        return float(metrics["micro"]["f1"])
    if key == "exact_match":
        return float(metrics["exact_match_accuracy"])
    if key == "parse_success":
        return float(metrics["parse_success_rate"])
    raise KeyError(key)


def load_metrics(path: Path) -> Dict[str, Any]:
    """读取 JSON 指标文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def save_summary(summary: Dict[str, Any], output_json: Path, output_md: Path) -> None:
    """同时写出 JSON 和 Markdown 摘要。"""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qwen3-8B LoRA Variant Benchmark",
        "",
        "| Variant | Validation F1 | Validation EM | Validation Parse | Test F1 | Test EM | Test Parse |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["ranked_variants"]:
        val_metrics = row.get("validation_metrics")
        test_metrics = row.get("test_metrics")
        lines.append(
            "| {variant} | {val_f1} | {val_em} | {val_parse} | {test_f1} | {test_em} | {test_parse} |".format(
                variant=row["variant"],
                val_f1=f"{metric_cell(val_metrics, 'micro_f1'):.4f}" if val_metrics else "-",
                val_em=f"{metric_cell(val_metrics, 'exact_match'):.4f}" if val_metrics else "-",
                val_parse=f"{metric_cell(val_metrics, 'parse_success'):.4f}" if val_metrics else "-",
                test_f1=f"{metric_cell(test_metrics, 'micro_f1'):.4f}" if test_metrics else "-",
                test_em=f"{metric_cell(test_metrics, 'exact_match'):.4f}" if test_metrics else "-",
                test_parse=f"{metric_cell(test_metrics, 'parse_success'):.4f}" if test_metrics else "-",
            )
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """按配置批量跑 LoRA 变体实验。"""
    args = parse_args()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    python_executable = Path(sys.executable)
    if Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve():
        raise RuntimeError(
            f"Please run this script with the repository virtualenv python. "
            f"sys.prefix={Path(sys.prefix).resolve()} expected={(PROJECT_ROOT / '.venv').resolve()}"
        )

    prompt_path = (PROJECT_ROOT / DEFAULT_PROMPT).resolve()
    validation_path = (PROJECT_ROOT / AUGMENT_DIR / "merged_chatml_validation.jsonl").resolve()
    test_path = (PROJECT_ROOT / AUGMENT_DIR / "merged_chatml_test.jsonl").resolve()

    summary_rows: List[Dict[str, Any]] = []
    for variant in args.variants:
        source_config_path = (PROJECT_ROOT / VARIANT_CONFIGS[variant]).resolve()
        config = load_training_config(source_config_path)
        training_prompt_path = config.get("system_prompt_path")
        if training_prompt_path is None or training_prompt_path.resolve() != prompt_path:
            raise RuntimeError(f"{variant} prompt mismatch: {training_prompt_path} != {prompt_path}")
        if bool(config.get("enable_thinking", False)):
            raise RuntimeError(f"{variant} must keep enable_thinking=false for this benchmark.")

        runtime_output_dir = (results_dir / "outputs" / variant).resolve()
        config["output_dir"] = runtime_output_dir
        if args.num_train_epochs is not None:
            config["num_train_epochs"] = float(args.num_train_epochs)
        # 这个 benchmark 只比较完整训练后的最终 adapter，因此跳过训练中的 eval/checkpoint，
        # 以控制变体矩阵的总时长，避免在重复验证上消耗数小时。
        config["eval_strategy"] = "no"
        config["save_strategy"] = "no"
        config["load_best_model_at_end"] = False
        config["logging_steps"] = max(20, int(config.get("logging_steps", 20)))
        runtime_config_path = materialize_runtime_config(
            config,
            results_dir / "runtime_configs" / f"{variant}.yaml",
        )

        variant_row: Dict[str, Any] = {
            "variant": variant,
            "config_path": str(runtime_config_path),
            "source_config_path": str(source_config_path),
            "output_dir": str(config["output_dir"]),
            "training_prompt_path": str(training_prompt_path),
            "validation_metrics": None,
            "test_metrics": None,
        }

        if not args.skip_train:
            print(f"[{variant}] train -> {runtime_output_dir}")
            train_command = [
                str(python_executable),
                "scripts/train/train_finetune.py",
                "--config",
                str(runtime_config_path),
                "--do-train",
                "--dry-run-samples",
                "0",
                "--disable-thinking",
            ]
            if args.max_train_samples is not None:
                train_command.extend(["--max-train-samples", str(args.max_train_samples)])
            if args.max_eval_samples is not None:
                train_command.extend(["--max-eval-samples", str(args.max_eval_samples)])
            run_command(train_command, results_dir / "logs" / f"{variant}_train.log")

        adapter_path = (config["output_dir"] / "final_adapter").resolve()
        if not adapter_path.exists():
            raise RuntimeError(f"{variant} adapter not found after training: {adapter_path}")

        if not args.skip_validation:
            print(f"[{variant}] validation -> {adapter_path}")
            validation_prefix = results_dir / f"{variant}_validation"
            predict_command = [
                str(python_executable),
                "scripts/inference/predict.py",
                "--config",
                DEFAULT_INFER_CONFIG,
                "--backend",
                args.backend,
                "--input-path",
                str(validation_path),
                "--adapter-path",
                str(adapter_path),
                "--system-prompt-path",
                str(prompt_path),
                "--output-path",
                str(validation_prefix.with_name(validation_prefix.name + "_predictions.jsonl")),
                "--metrics-path",
                str(validation_prefix.with_name(validation_prefix.name + "_metrics.txt")),
                "--metrics-json-path",
                str(validation_prefix.with_name(validation_prefix.name + "_metrics.json")),
                "--disable-thinking",
            ]
            if args.prediction_limit is not None:
                predict_command.extend(["--limit", str(args.prediction_limit)])
            run_command(predict_command, results_dir / "logs" / f"{variant}_validation.log")
            variant_row["validation_metrics"] = load_metrics(
                validation_prefix.with_name(validation_prefix.name + "_metrics.json")
            )

        if not args.skip_test:
            print(f"[{variant}] test -> {adapter_path}")
            test_prefix = results_dir / f"{variant}_test"
            predict_command = [
                str(python_executable),
                "scripts/inference/predict.py",
                "--config",
                DEFAULT_INFER_CONFIG,
                "--backend",
                args.backend,
                "--input-path",
                str(test_path),
                "--adapter-path",
                str(adapter_path),
                "--system-prompt-path",
                str(prompt_path),
                "--output-path",
                str(test_prefix.with_name(test_prefix.name + "_predictions.jsonl")),
                "--metrics-path",
                str(test_prefix.with_name(test_prefix.name + "_metrics.txt")),
                "--metrics-json-path",
                str(test_prefix.with_name(test_prefix.name + "_metrics.json")),
                "--disable-thinking",
            ]
            if args.prediction_limit is not None:
                predict_command.extend(["--limit", str(args.prediction_limit)])
            run_command(predict_command, results_dir / "logs" / f"{variant}_test.log")
            variant_row["test_metrics"] = load_metrics(test_prefix.with_name(test_prefix.name + "_metrics.json"))

        summary_rows.append(variant_row)

    ranked_rows = sorted(
        summary_rows,
        key=lambda row: (
            metric_cell(row["validation_metrics"], "micro_f1") if row["validation_metrics"] else -1.0,
            metric_cell(row["validation_metrics"], "exact_match") if row["validation_metrics"] else -1.0,
            metric_cell(row["validation_metrics"], "parse_success") if row["validation_metrics"] else -1.0,
        ),
        reverse=True,
    )
    summary = {
        "backend": args.backend,
        "prompt_path": str(prompt_path),
        "variants": summary_rows,
        "ranked_variants": ranked_rows,
    }
    save_summary(summary, results_dir / "summary.json", results_dir / "summary.md")


if __name__ == "__main__":
    main()
