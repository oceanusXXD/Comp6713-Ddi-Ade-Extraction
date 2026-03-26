import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import Trainer, TrainingArguments

from src.data_utils import (
    SupervisedDataCollator,
    build_supervised_dataset,
    compute_dataset_statistics,
    summarize_chat_dataset,
)
from src.model_utils import build_model_and_tokenizer, load_tokenizer, load_training_config, save_adapter_artifacts
from src.observability import (
    JsonlMetricsCallback,
    collect_parameter_statistics,
    collect_runtime_environment,
    write_json,
)
from src.prompting import load_system_prompt

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen-style chat models on ADE/DDI extraction.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_qwen3_8b_lora.yaml",
        help="Path to the training YAML config.",
    )
    parser.add_argument(
        "--do-train",
        action="store_true",
        help="Run training. If omitted, the script validates config, loads the tokenizer, and encodes a sample.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for train samples, useful for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap for validation samples, useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run-samples",
        type=int,
        default=8,
        help="How many samples to encode during the default dry run.",
    )
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_resume_path(resume_from_checkpoint: Optional[str]) -> Optional[Path]:
    if not resume_from_checkpoint:
        return None
    return Path(resume_from_checkpoint).expanduser().resolve()


def build_training_arguments(config):
    return TrainingArguments(
        output_dir=str(config["output_dir"]),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        logging_steps=config["logging_steps"],
        eval_strategy=config["eval_strategy"],
        eval_steps=config["eval_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        gradient_checkpointing=config["gradient_checkpointing"],
        remove_unused_columns=config["remove_unused_columns"],
        report_to=config["report_to"],
        bf16=config["bf16"],
        fp16=config["fp16"],
        max_grad_norm=config["max_grad_norm"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config.get("metric_for_best_model"),
        greater_is_better=config.get("greater_is_better"),
        label_names=["labels"],
        seed=config["seed"],
        data_seed=config["seed"],
    )


def log_dataset_summary(config) -> None:
    train_summary = summarize_chat_dataset(config["train_path"])
    LOGGER.info("Train summary: %s", train_summary)

    validation_path = config.get("validation_path")
    if validation_path is not None:
        validation_summary = summarize_chat_dataset(validation_path)
        LOGGER.info("Validation summary: %s", validation_summary)


def build_encoded_preview(config, tokenizer, system_prompt: str, dry_run_samples: int) -> None:
    if dry_run_samples <= 0:
        LOGGER.info("Dry-run sample encoding skipped because --dry-run-samples=%s", dry_run_samples)
        return

    train_preview, train_stats = build_supervised_dataset(
        config["train_path"],
        tokenizer,
        system_prompt=system_prompt,
        max_length=config["max_seq_length"],
        enable_thinking=config.get("enable_thinking"),
        limit=dry_run_samples,
    )
    LOGGER.info("Dry-run train encoding stats: %s", train_stats)
    if len(train_preview) > 0:
        LOGGER.info(
            "Dry-run first train sample lengths: input_ids=%s labels=%s",
            len(train_preview[0]["input_ids"]),
            len(train_preview[0]["labels"]),
        )

    validation_path = config.get("validation_path")
    if validation_path is not None:
        eval_preview, eval_stats = build_supervised_dataset(
            validation_path,
            tokenizer,
            system_prompt=system_prompt,
            max_length=config["max_seq_length"],
            enable_thinking=config.get("enable_thinking"),
            limit=min(dry_run_samples, 4),
        )
        LOGGER.info("Dry-run validation encoding stats: %s", eval_stats)
        if len(eval_preview) > 0:
            LOGGER.info(
                "Dry-run first validation sample lengths: input_ids=%s labels=%s",
                len(eval_preview[0]["input_ids"]),
                len(eval_preview[0]["labels"]),
            )


def observability_dir(config) -> Path:
    return config["output_dir"] / "observability"


def save_dataset_statistics(config, tokenizer, system_prompt: str) -> None:
    output_dir = observability_dir(config)
    train_stats = compute_dataset_statistics(
        config["train_path"],
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_length=config["max_seq_length"],
        enable_thinking=config.get("enable_thinking"),
    )
    write_json(output_dir / "train_dataset_stats.json", train_stats)
    LOGGER.info("Saved train dataset statistics to %s", output_dir / "train_dataset_stats.json")

    validation_path = config.get("validation_path")
    if validation_path is not None:
        validation_stats = compute_dataset_statistics(
            validation_path,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_length=config["max_seq_length"],
            enable_thinking=config.get("enable_thinking"),
        )
        write_json(output_dir / "validation_dataset_stats.json", validation_stats)
        LOGGER.info("Saved validation dataset statistics to %s", output_dir / "validation_dataset_stats.json")


def save_run_configuration(config) -> None:
    output_dir = observability_dir(config)
    write_json(output_dir / "runtime_environment.json", collect_runtime_environment())
    write_json(output_dir / "training_config_snapshot.json", config)
    LOGGER.info("Saved runtime environment and config snapshot under %s", output_dir)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_training_config(args.config)
    if args.enable_thinking and args.disable_thinking:
        raise ValueError("Use at most one of --enable-thinking or --disable-thinking.")
    if args.enable_thinking:
        config["enable_thinking"] = True
    elif args.disable_thinking:
        config["enable_thinking"] = False

    cli_resume = resolve_resume_path(args.resume_from_checkpoint)
    if cli_resume is not None:
        config["resume_from_checkpoint"] = cli_resume

    set_global_seed(config["seed"])
    LOGGER.info("Loaded config from %s", config["config_path"])
    log_dataset_summary(config)
    LOGGER.info(
        "Training config: model=%s finetune_type=%s load_in_4bit=%s max_seq_length=%s output_dir=%s",
        config["model_name_or_path"],
        config["finetune_type"],
        config["load_in_4bit"],
        config["max_seq_length"],
        config["output_dir"],
    )

    system_prompt = load_system_prompt(
        str(config["system_prompt_path"]) if config.get("system_prompt_path") is not None else None
    )
    tokenizer = load_tokenizer(config)
    LOGGER.info("Loaded tokenizer for %s", config["model_name_or_path"])
    save_run_configuration(config)
    save_dataset_statistics(config, tokenizer, system_prompt)
    build_encoded_preview(config, tokenizer, system_prompt, args.dry_run_samples)

    if not args.do_train:
        LOGGER.info("Dry run complete. Pass --do-train to load the model and start training.")
        return

    max_seq_length = config["max_seq_length"]
    enable_thinking = config.get("enable_thinking")

    train_dataset, train_stats = build_supervised_dataset(
        config["train_path"],
        tokenizer,
        system_prompt=system_prompt,
        max_length=max_seq_length,
        enable_thinking=enable_thinking,
        limit=args.max_train_samples,
    )
    LOGGER.info("Prepared train dataset: %s", train_stats)

    eval_dataset = None
    validation_path = config.get("validation_path")
    if validation_path is not None:
        eval_dataset, eval_stats = build_supervised_dataset(
            validation_path,
            tokenizer,
            system_prompt=system_prompt,
            max_length=max_seq_length,
            enable_thinking=enable_thinking,
            limit=args.max_eval_samples,
        )
        LOGGER.info("Prepared validation dataset: %s", eval_stats)

    model, tokenizer = build_model_and_tokenizer(config, tokenizer=tokenizer)
    parameter_stats = collect_parameter_statistics(model)
    write_json(observability_dir(config) / "parameter_stats.json", parameter_stats)
    LOGGER.info("Model parameter statistics: %s", parameter_stats)
    training_arguments = build_training_arguments(config)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=[JsonlMetricsCallback(observability_dir(config) / "training_metrics.jsonl")],
    )

    resume_from_checkpoint = config.get("resume_from_checkpoint")
    if resume_from_checkpoint is not None:
        LOGGER.info("Resuming from checkpoint: %s", resume_from_checkpoint)

    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
    )
    trainer.save_state()
    write_json(observability_dir(config) / "trainer_state_summary.json", trainer.state.log_history)
    trainer.save_model()
    adapter_dir = save_adapter_artifacts(model, tokenizer, config["output_dir"])
    LOGGER.info("Saved final adapter artifacts to %s", adapter_dir)


if __name__ == "__main__":
    main()
