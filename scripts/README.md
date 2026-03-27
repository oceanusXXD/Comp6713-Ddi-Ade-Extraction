# 脚本目录说明

这个目录放的是仓库的命令行入口脚本。大部分日常操作都从这里进入。

## 文件和子目录说明

### `train/`

- `train/train_finetune.py`
  主训练脚本。负责读取训练配置、加载模型和 tokenizer、执行 dry run、正式训练、保存 checkpoint 和最终 adapter。

### `inference/`

- `inference/predict.py`
  主推理脚本。支持批量数据推理、单条文本推理、切换 `transformers` / `vllm` 后端、输出预测文件和指标文件。

### `evaluation/`

- `evaluation/evaluate_predictions.py`
  主评估脚本。对预测文件做解析、规范化和指标计算，输出 `.txt` 和 `.json` 结果。
- `evaluation/evaluate_predictions_by_augmentation.py`
  按 `augmentation_type` 分组评估预测结果，适合检查 `paraphrase`、`negative`、`hardcase`、`margincase` 各类增强样本的表现。

### `analysis/`

- `analysis/analyze_dataset.py`
  数据统计脚本，用来分析长度分布、token 统计和样本结构。
- `analysis/audit_and_prepare_final_dataset.py`
  数据审计与物化脚本，用来清洗数据、排查 split 污染、合并增强样本，并生成当前主线推荐数据版本。
- `analysis/fetch_evaluate_datasets.py`
  评测数据整理脚本，用来下载公开评测集、解压压缩包、生成 `evaluate_datasets/MANIFEST.json`，并构建 `seen_style_core/` 镜像。

### `experiments/`

- `experiments/run_qwen3_lora_variant_benchmark.py`
  变体实验脚本，用来批量训练并评估不同 LoRA 方案，自动写运行时配置、日志、预测结果和实验摘要。

## 最常用命令

### 重新生成推荐训练数据

```bash
.venv/bin/python scripts/analysis/audit_and_prepare_final_dataset.py
```

### 训练当前主线配置

```bash
.venv/bin/python scripts/train/train_finetune.py \
  --config configs/qwen3_8b_lora_ddi_ade_final.yaml \
  --do-train
```

### 对当前验证集推理

```bash
.venv/bin/python scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_final.yaml
```

### 评估预测结果

```bash
.venv/bin/python scripts/evaluation/evaluate_predictions.py \
  --predictions-path results/your_predictions.jsonl
```

### 按增强类型做评估

```bash
.venv/bin/python scripts/evaluation/evaluate_predictions_by_augmentation.py \
  --predictions-path results/your_predictions.jsonl \
  --source-path data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train_augmentations.jsonl
```

## 你应该怎么理解这些脚本

- 想训练模型：
  先看 `analysis/audit_and_prepare_final_dataset.py`，再看 `train/train_finetune.py`
- 想跑模型：
  看 `inference/predict.py`
- 想算指标：
  看 `evaluation/evaluate_predictions.py`
- 想看数据质量：
  看 `analysis/analyze_dataset.py`
- 想补外部评测：
  看 `analysis/fetch_evaluate_datasets.py`
- 想批量对比不同训练变体：
  看 `experiments/run_qwen3_lora_variant_benchmark.py`
