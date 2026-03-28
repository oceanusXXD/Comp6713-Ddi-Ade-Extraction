# 配置目录说明

这个目录保存当前维护的训练与推理配置文件。

## 当前保留文件

- `qwen3_8b_lora_ddi_ade_final.yaml`
  当前主线训练配置。
- `infer_qwen3_8b_lora_ddi_ade_final.yaml`
  当前主线推理配置。

## 两份配置如何配合

- `scripts/train/train_finetune.py`
  读取 `qwen3_8b_lora_ddi_ade_final.yaml`
- `scripts/inference/predict.py`
  读取 `infer_qwen3_8b_lora_ddi_ade_final.yaml`

训练配置决定：

- 基座模型路径
- 系统 prompt 路径
- 训练集和验证集路径
- LoRA / rsLoRA 参数
- 训练超参数
- 输出目录

推理配置决定：

- 基座模型与 adapter 路径
- 推理输入文件
- 推理后端
- 生成参数
- 预测和指标输出路径

## 当前主线实际指向

### 训练配置

- 基座模型：`../models/Qwen3-8B`
- 训练集：`data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`
- 验证集：`data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- 输出目录：`outputs/qwen3_8b_lora_ddi_ade_final_aug_e4`

### 推理配置

- 基座模型：`../models/Qwen3-8B`
- adapter：`outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/final_adapter`
- 默认输入：`data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- 默认输出：
  - `results/inference_runs/qwen3_8b_lora_ddi_ade_final_aug_e4_validation_predictions.jsonl`
  - `results/inference_runs/qwen3_8b_lora_ddi_ade_final_aug_e4_validation_metrics.txt`

## 最常改的字段

### 训练配置里

- `train_path`
- `validation_path`
- `model_name_or_path`
- `output_dir`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- `use_rslora`
- `learning_rate`
- `num_train_epochs`

### 推理配置里

- `model.base_model_name_or_path`
- `model.adapter_path`
- `data.input_path`
- `backend`
- `inference.batch_size`
- `inference.max_new_tokens`
- `output.predictions_path`
- `output.metrics_path`

## 使用建议

- 想继续当前主线：只改路径和少量超参数，不要先改目录名。
- 想开新实验：复制现有配置，再在文件名里显式写新 tag。
- 改完配置后，优先做一次 `train_finetune.py --dry-run-samples 8` 或小样本推理验证。
