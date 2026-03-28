# 输出目录说明

这个目录保存训练产物、LoRA adapter、checkpoint、可观测性文件和规则基线输出。

## 当前最重要的目录

如果你只关心当前推荐训练产物，优先看：

- `qwen3_8b_lora_ddi_ade_final_aug_e4/`

这个目录对应当前主线训练配置 `configs/qwen3_8b_lora_ddi_ade_final.yaml` 中的 `output_dir`。

目录名可按下面方式理解：

- `qwen3_8b`
  基座模型族
- `lora`
  微调方式
- `ddi_ade`
  任务范围
- `final_aug_e4`
  数据 / 实验版本标记

## 当前保留的主要子目录

- `baseline/`
  规则基线输出。
- `qwen3_8b_lora/`
  较早一轮保留的可分享 adapter 目录。
- `qwen3_8b_lora_ddi_ade_c5fc8c06/`
  基于旧数据版本 `c5fc8c06` 的历史训练产物。
- `qwen3_8b_lora_ddi_ade_final/`
  一轮较早 `final` 版本训练留下的输出目录。
- `qwen3_8b_lora_ddi_ade_final_aug_e4/`
  当前主线训练输出目录。
- `r4_lora_variant_baseline_results.txt`
  已清理历史 r4 变体目录后保留下来的摘要结果。

## `qwen3_8b_lora_ddi_ade_final_aug_e4/` 怎么看

这个目录里重点关注：

- `checkpoint-620/`
  2026-03-27 benchmark 中 DDI 迁移表现最强的 rsLoRA checkpoint。
- `checkpoint-1085/`
  训练后段保留的中间 checkpoint，主要用于回看训练过程。
- `checkpoint-1232/`
  本轮训练完成时的最终 checkpoint。
- `final_adapter/`
  训练完成后导出的最终可加载 adapter。
- `observability/`
  配置快照、参数统计、训练日志和环境信息。

## 与 2026-03-27 benchmark 的关系

如果你是从 `results/benchmark_suite_vllm_batch64_20260327/` 倒查模型来源：

- `rslora_620`
  对应这里的 `checkpoint-620/`
- `rslora_1232`
  对应这里的 `checkpoint-1232/`
- `rslora_930`
  只保留了 benchmark 结果快照，没有在 `outputs/` 下继续单独保留 `checkpoint-930/`

## 典型训练输出目录的内容

LoRA 训练目录通常包含：

- `final_adapter/`
- `checkpoint-*`
- `observability/`
- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `chat_template.jinja`
- `added_tokens.json`
- `merges.txt`
- `vocab.json`

## `observability/` 常见文件

- `runtime_environment.json`
- `training_config_snapshot.json`
- `train_dataset_stats.json`
- `validation_dataset_stats.json`
- `parameter_stats.json`
- `training_metrics.jsonl`
- `overfit_watch.log`

## 规则基线输出

`baseline/` 下主要保留：

- `best_config.json`
- `baseline_validation_preds.jsonl`
- `baseline_validation_preds_metrics.txt`
- `baseline_test_preds.jsonl`
- `baseline_test_preds_metrics.txt`

## 注意事项

- 这里有较多历史实验产物，不能只凭目录名判断是否为当前推荐版本。
- 判断“主线输出目录”时，应以 `configs/` 中当前训练配置引用的 `output_dir` 为准。
- 大体积权重文件通常通过 Git LFS 跟踪。
