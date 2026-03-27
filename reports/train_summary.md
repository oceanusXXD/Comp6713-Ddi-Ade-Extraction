# 训练总结

## 当前主线

- 主线训练配置：`configs/qwen3_8b_lora_ddi_ade_final.yaml`
- 主线推理配置：`configs/infer_qwen3_8b_lora_ddi_ade_final.yaml`
- 主线训练数据：`data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`
- 主线验证数据：`data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- 主线测试数据：`data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_test.jsonl`
- 主线增强 sidecar：`data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train_augmentations.jsonl`

## 数据规模

- 训练集行数：`4924`
- 验证集行数：`436`
- 测试集行数：`619`
- 增强 sidecar 行数：`1012`
- 增强类型分布：
  - `negative = 320`
  - `margincase = 265`
  - `hardcase = 247`
  - `paraphrase = 180`

## 当前训练参数

- 基座模型：`/home/coder/data/models/Qwen3-8B`
- 微调方式：LoRA
- 变体：`rsLoRA + all-linear`
- `lora_r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.05`
- `per_device_train_batch_size = 4`
- `gradient_accumulation_steps = 4`
- `learning_rate = 3.5e-5`
- `num_train_epochs = 4`
- `output_dir = outputs/qwen3_8b_lora_ddi_ade_final_aug_e4`

## 当前运行状态

- dry run：已通过
- 正式训练：已启动
- 训练输出目录：`outputs/qwen3_8b_lora_ddi_ade_final_aug_e4`
- 观测文件：`outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/observability/training_metrics.jsonl`
- 已启用验证损失监控，若后续连续验证变差，会自动中断训练
