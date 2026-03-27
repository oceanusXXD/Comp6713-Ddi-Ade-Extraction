# 输出目录说明

这个目录保存训练产物、LoRA adapter、checkpoint、可观测性文件和规则基线输出。

## 目录里的主要子目录是干什么的

- `baseline/`
  规则基线的输出目录，包含最佳规则配置、验证集 / 测试集预测和对应指标。
- `qwen3_8b_lora/`
  较早一轮保留的可分享 LoRA adapter 目录，里面有 `final_adapter/` 和 `observability/`。
- `qwen3_8b_lora_ddi_ade_c5fc8c06/`
  一套基于 `c5fc8c06` 数据版本跑出的历史训练产物，含 adapter、checkpoint 和观测文件。
- `qwen3_8b_lora_ddi_ade_final/`
  一轮较早 `final` 版本训练留下的输出目录。
- `qwen3_8b_lora_ddi_ade_final_aug_e4/`
  当前主线训练输出目录，现阶段主要保存 checkpoint 和 `observability/`。
- `r4_lora_variant_baseline_results.txt`
  已清理的 r4 系列 LoRA 变体实验摘要，只保留可追溯的基础结果，不保留原始目录。

## 已清理的历史实验

以下 r4 变体目录已经从仓库中删除，仅保留摘要文件 `r4_lora_variant_baseline_results.txt`：

- `qwen3_8b_lora_ddi_ade_r4/`
- `qwen3_8b_lora_ddi_ade_r4_dora/`
- `qwen3_8b_lora_ddi_ade_r4_lorafa/`
- `qwen3_8b_lora_ddi_ade_r4_loraplus/`
- `qwen3_8b_lora_ddi_ade_r4_pissa16/`
- `qwen3_8b_lora_ddi_ade_r4_rslora/`

## 一个典型训练输出目录里会有什么

以 LoRA 训练目录为例，常见文件包括：

- `adapter_model.safetensors`
  LoRA 权重本体。
- `adapter_config.json`
  LoRA 配置。
- `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`
  tokenizer 配置相关文件。
- `chat_template.jinja`
  chat template。
- `added_tokens.json`
  额外 token 定义。
- `merges.txt`、`vocab.json`
  tokenizer 词表相关文件。
- `final_adapter/`
  最终可直接加载的 adapter 目录。
- `checkpoint-*`
  训练中间 checkpoint。
- `observability/`
  运行环境、配置快照、数据统计、训练日志等观测文件。

## `observability/` 里的文件一般怎么理解

常见文件包括：

- `runtime_environment.json`
  运行环境和依赖信息快照。
- `training_config_snapshot.json`
  训练配置快照。
- `train_dataset_stats.json`
  训练集统计。
- `validation_dataset_stats.json`
  验证集统计。
- `parameter_stats.json`
  参数统计。
- `training_metrics.jsonl`
  训练过程指标日志。
- `overfit_watch.log`
  过拟合观察日志。

## 当前应该重点看哪个目录

如果你要看当前主线训练，优先看：

- `qwen3_8b_lora_ddi_ade_final_aug_e4/`

如果你要看一个已保留的、可以直接分享的 adapter 示例，优先看：

- `qwen3_8b_lora/final_adapter/`

它的详细文件说明见：

- `qwen3_8b_lora/final_adapter/README.md`

## 规则基线输出文件说明

`baseline/` 下当前已有：

- `best_config.json`
  验证集上挑出的最佳规则参数。
- `baseline_validation_preds.jsonl`
  基线在验证集上的预测。
- `baseline_validation_preds_metrics.txt`
  验证集指标。
- `baseline_test_preds.jsonl`
  基线在测试集上的预测。
- `baseline_test_preds_metrics.txt`
  测试集指标。

## 注意事项

- 这里的很多目录是历史实验产物，不一定代表当前推荐版本。
- 判断“当前主线”要以 `configs/` 下最新配置引用的 `output_dir` 为准。
- 大体积权重文件通常通过 Git LFS 跟踪。
