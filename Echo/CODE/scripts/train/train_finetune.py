"""仓库主训练入口。

这个脚本负责串起完整训练流程：
- 读取训练配置
- 加载 tokenizer 与模型
- 统计数据集
- 做 dry run 编码检查
- 构建 Trainer 并执行训练
- 保存最终 adapter 和观测文件
"""

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
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from torch.utils.data import WeightedRandomSampler

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


def unwrap_module(model):
    """从可能被多层 wrapper 包裹的模型中拿到最内层真实模型。"""
    candidate = model
    while hasattr(candidate, "module"):
        candidate = candidate.module
    return candidate


class WeightedSamplingTrainer(Trainer):
    """在 Hugging Face Trainer 基础上增加两类能力：

    1. 按样本权重做加权采样。
    2. 按样本权重和 label smoothing 自定义 loss 计算。
    """
    def __init__(
        self,
        *args,
        enable_weighted_sampling: bool = False,
        use_sample_loss_weights: bool = False,
        label_smoothing_factor: float = 0.0,
        adapter_optimizer: str = "default",
        loraplus_lr_ratio: float = 16.0,
        loraplus_lr_embedding: float = 1e-6,
        loraplus_weight_decay: Optional[float] = None,
        lora_r: int = 8,
        lora_alpha: int = 8,
        use_rslora: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_weighted_sampling = enable_weighted_sampling
        self.use_sample_loss_weights = use_sample_loss_weights
        self.label_smoothing_factor = float(label_smoothing_factor)
        self.adapter_optimizer = str(adapter_optimizer).lower()
        self.loraplus_lr_ratio = float(loraplus_lr_ratio)
        self.loraplus_lr_embedding = float(loraplus_lr_embedding)
        self.loraplus_weight_decay = loraplus_weight_decay
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.use_rslora = bool(use_rslora)

    def _get_train_sampler(self, train_dataset=None):
        """在单进程训练下启用 `WeightedRandomSampler`。"""
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if not self.enable_weighted_sampling or dataset is None:
            return super()._get_train_sampler(train_dataset)
        sample_weights = getattr(dataset, "sample_weights", None)
        if not sample_weights:
            return super()._get_train_sampler(train_dataset)
        if self.args.world_size > 1:
            LOGGER.warning("Weighted sampling is only enabled for single-process training; falling back to default.")
            return super()._get_train_sampler(train_dataset)
        generator = torch.Generator()
        generator.manual_seed(self.args.data_seed if self.args.data_seed is not None else self.args.seed)
        return WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """覆盖默认 loss 计算，支持样本级 loss weighting。"""
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights", None)
        outputs = model(**inputs)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("Model output does not contain logits; cannot compute custom training loss.")

        # 这里沿用 Causal LM 的标准“右移一位”监督方式。
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        per_token_loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
            ignore_index=-100,
            label_smoothing=self.label_smoothing_factor,
        ).view_as(shift_labels)
        valid_mask = shift_labels.ne(-100)
        valid_mask_f = valid_mask.to(per_token_loss.dtype)
        per_example_loss = (per_token_loss * valid_mask_f).sum(dim=1) / valid_mask_f.sum(dim=1).clamp_min(1.0)

        if self.use_sample_loss_weights and loss_weights is not None:
            # 按样本权重聚合 loss，避免少数关键类型被大批简单样本淹没。
            normalized_weights = loss_weights.to(per_example_loss.device, dtype=per_example_loss.dtype)
            loss = (per_example_loss * normalized_weights).sum() / normalized_weights.sum().clamp_min(1e-8)
        else:
            loss = per_example_loss.mean()

        if return_outputs:
            return loss, outputs
        return loss

    def create_optimizer(self):
        """在默认优化器之外，额外支持 LoRA+ 和 LoRA-FA 两类 adapter 优化器。"""
        if self.optimizer is not None:
            return self.optimizer
        if self.adapter_optimizer == "default":
            return super().create_optimizer()

        try:
            from peft import PeftModel
            from peft.optimizers import create_lorafa_optimizer, create_loraplus_optimizer
        except ImportError as exc:
            raise ImportError("Adapter optimizer variants require peft with optimizer support.") from exc

        opt_model = unwrap_module(self.model)
        if not isinstance(opt_model, PeftModel):
            raise ValueError(
                f"adapter_optimizer={self.adapter_optimizer} requires a PEFT-wrapped model, got {type(opt_model)!r}."
            )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        unsupported_optimizer_keys = {"params", "model", "optimizer_dict"} & set(optimizer_kwargs)
        if unsupported_optimizer_keys:
            raise ValueError(
                f"adapter_optimizer={self.adapter_optimizer} is not compatible with optimizer kwargs "
                f"{sorted(unsupported_optimizer_keys)}."
            )

        learning_rate = optimizer_kwargs.pop("lr", self.args.learning_rate)
        weight_decay = optimizer_kwargs.pop("weight_decay", self.args.weight_decay)

        if self.adapter_optimizer == "loraplus":
            loraplus_weight_decay = self.loraplus_weight_decay
            if loraplus_weight_decay is None:
                loraplus_weight_decay = weight_decay
            self.optimizer = create_loraplus_optimizer(
                opt_model,
                optimizer_cls,
                lr=learning_rate,
                loraplus_lr_ratio=self.loraplus_lr_ratio,
                loraplus_lr_embedding=self.loraplus_lr_embedding,
                loraplus_weight_decay=loraplus_weight_decay,
                **optimizer_kwargs,
            )
            return self.optimizer

        if optimizer_kwargs:
            LOGGER.warning("LoRA-FA ignores optimizer kwargs from TrainingArguments: %s", sorted(optimizer_kwargs))
        self.optimizer = create_lorafa_optimizer(
            opt_model,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lr=learning_rate,
            weight_decay=weight_decay,
            use_rslora=self.use_rslora,
        )
        return self.optimizer


def parse_args() -> argparse.Namespace:
    """解析训练命令行参数。"""
    parser = argparse.ArgumentParser(description="Fine-tune Qwen-style chat models on ADE/DDI extraction.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml",
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
    """初始化脚本日志格式。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_global_seed(seed: int) -> None:
    """统一设置 Python / Torch 随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_resume_path(resume_from_checkpoint: Optional[str]) -> Optional[Path]:
    """把 resume 路径解析成绝对路径。"""
    if not resume_from_checkpoint:
        return None
    return Path(resume_from_checkpoint).expanduser().resolve()


def build_training_arguments(config):
    """把仓库训练配置映射成 `TrainingArguments`。"""
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
        label_smoothing_factor=config.get("label_smoothing_factor", 0.0),
    )


def log_dataset_summary(config) -> None:
    """把训练集和验证集的基础摘要打到日志。"""
    train_summary = summarize_chat_dataset(config["train_path"])
    LOGGER.info("Train summary: %s", train_summary)

    validation_path = config.get("validation_path")
    if validation_path is not None:
        validation_summary = summarize_chat_dataset(validation_path)
        LOGGER.info("Validation summary: %s", validation_summary)


def build_encoded_preview(config, tokenizer, system_prompt: str, dry_run_samples: int) -> None:
    """在真正加载大模型前，先做小样本编码预览。"""
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
    """返回当前训练运行对应的 observability 目录。"""
    return config["output_dir"] / "observability"


def save_dataset_statistics(config, tokenizer, system_prompt: str) -> None:
    """把训练 / 验证集统计结果写入 observability 目录。"""
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
    """记录运行环境与训练配置快照。"""
    output_dir = observability_dir(config)
    write_json(output_dir / "runtime_environment.json", collect_runtime_environment())
    write_json(output_dir / "training_config_snapshot.json", config)
    LOGGER.info("Saved runtime environment and config snapshot under %s", output_dir)


def main() -> None:
    """训练脚本主流程。"""
    configure_logging()
    args = parse_args()
    if Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve():
        raise RuntimeError(
            f"Please run training with the repository virtualenv python. "
            f"sys.prefix={Path(sys.prefix).resolve()} expected={(PROJECT_ROOT / '.venv').resolve()}"
        )
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
    LOGGER.info(
        "Adapter config: use_rslora=%s use_dora=%s init_lora_weights=%s adapter_optimizer=%s",
        config.get("use_rslora", False),
        config.get("use_dora", False),
        config.get("init_lora_weights", True),
        config.get("adapter_optimizer", "default"),
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

    # 训练集和验证集都在这里编码成 Trainer 可直接消费的数据结构。
    train_dataset, train_stats = build_supervised_dataset(
        config["train_path"],
        tokenizer,
        system_prompt=system_prompt,
        max_length=max_seq_length,
        enable_thinking=enable_thinking,
        limit=args.max_train_samples,
        empty_target_sampling_weight=config.get("empty_target_sampling_weight", 1.0),
        ddi_sampling_weight=config.get("ddi_sampling_weight", 1.0),
        ddi_int_sampling_weight=config.get("ddi_int_sampling_weight", 1.0),
        multi_relation_sampling_weight=config.get("multi_relation_sampling_weight", 1.0),
        empty_target_loss_weight=config.get("empty_target_loss_weight", 1.0),
        ddi_loss_weight=config.get("ddi_loss_weight", 1.0),
        ddi_int_loss_weight=config.get("ddi_int_loss_weight", 1.0),
        multi_relation_loss_weight=config.get("multi_relation_loss_weight", 1.0),
    )
    LOGGER.info("Prepared train dataset: %s", train_stats)
    if getattr(train_dataset, "sample_weights", None):
        min_weight = min(train_dataset.sample_weights)
        max_weight = max(train_dataset.sample_weights)
        unique_weights = sorted(set(round(weight, 4) for weight in train_dataset.sample_weights))
        LOGGER.info(
            "Train sample weights: strategy=%s min=%.4f max=%.4f unique=%s",
            config.get("train_sampling_strategy"),
            min_weight,
            max_weight,
            unique_weights[:10],
        )

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

    # 只有在 dry run 通过后，才真正加载大模型，避免无谓地占 GPU 显存。
    model, tokenizer = build_model_and_tokenizer(config, tokenizer=tokenizer)
    parameter_stats = collect_parameter_statistics(model)
    write_json(observability_dir(config) / "parameter_stats.json", parameter_stats)
    LOGGER.info("Model parameter statistics: %s", parameter_stats)
    training_arguments = build_training_arguments(config)
    trainer = WeightedSamplingTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=[JsonlMetricsCallback(observability_dir(config) / "training_metrics.jsonl")],
        enable_weighted_sampling=config.get("train_sampling_strategy") == "weighted",
        use_sample_loss_weights=bool(config.get("use_sample_loss_weights", False)),
        label_smoothing_factor=config.get("label_smoothing_factor", 0.0),
        adapter_optimizer=config.get("adapter_optimizer", "default"),
        loraplus_lr_ratio=config.get("loraplus_lr_ratio", 16.0),
        loraplus_lr_embedding=config.get("loraplus_lr_embedding", 1e-6),
        loraplus_weight_decay=config.get("loraplus_weight_decay"),
        lora_r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 8),
        use_rslora=config.get("use_rslora", False),
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
