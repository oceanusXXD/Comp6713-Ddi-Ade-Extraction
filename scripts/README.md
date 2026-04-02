# 脚本目录说明

这个目录集中放置仓库的命令行入口脚本。日常训练、推理、评估和数据整理都从这里进入。

## 子目录一览

### `train/`

- `train_finetune.py`
  主训练入口。负责读取配置、加载模型与 tokenizer、编码检查、执行训练、保存 adapter 和观测文件。

### `inference/`

- `predict.py`
  主推理入口。支持批量数据推理与单文本推理，支持 `transformers` 和 `vllm` 两类后端。

### `evaluation/`

- `evaluate_predictions.py`
  通用预测评估脚本，计算文本报告和 JSON 指标。
- `evaluate_predictions_by_augmentation.py`
  按 `augmentation_type` 切分评估，适合分析增强样本表现。
- `run_benchmark_suite.py`
  统一跑整套 benchmark。

### `analysis/`

- `analyze_dataset.py`
  数据统计与结构分析。
- `audit_and_prepare_final_dataset.py`
  数据审计与主线数据物化脚本。
- `fetch_evaluate_datasets.py`
  下载公开评测集、写 `evaluate_datasets/MANIFEST.json`，并构建 `seen_style_core/` 镜像。

### `demo/`

- `gradio_demo.py`
  Gradio demo (English UI): direct text inference, optional examples, and two supported presets: base-only plus the repo-relative LoRA adapter; see `scripts/demo/README.md`.

### `experiments/`

- `run_qwen3_lora_variant_benchmark.py`
  批量训练与比较不同 LoRA 方案的实验脚本。

## 推荐理解顺序

- 想重建主线数据：先看 `analysis/audit_and_prepare_final_dataset.py`
- 想训练模型：再看 `train/train_finetune.py`
- 想跑推理：看 `inference/predict.py`
- 想算指标：看 `evaluation/evaluate_predictions.py`
- 想做整套 benchmark：看 `evaluation/run_benchmark_suite.py`
- 想补充外部评测：看 `analysis/fetch_evaluate_datasets.py`

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
  --predictions-path results/inference_runs/your_predictions.jsonl
```

### 按增强类型做评估

```bash
.venv/bin/python scripts/evaluation/evaluate_predictions_by_augmentation.py \
  --predictions-path results/inference_runs/your_predictions.jsonl \
  --source-path data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train_augmentations.jsonl
```

## 使用建议

- 主线维护优先走 `analysis -> train -> inference -> evaluation` 这条路径。
- 新脚本尽量放进已经存在的功能分组，不要再往 `scripts/` 顶层堆单文件。
- 输出路径、数据路径和 README 说明应同步维护，避免脚本入口和仓库说明脱节。
