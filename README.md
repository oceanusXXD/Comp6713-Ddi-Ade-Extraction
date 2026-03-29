# COMP6713 DDI/ADE 抽取仓库

这个仓库用于医疗关系抽取，目标是从一段医学文本中抽取两类关系：

- `ADE`：药物导致的不良反应
- `DDI-*`：药物与药物之间的相互作用

当前维护主线基于 `Qwen3-8B` + rsLoRA，任务形式是结构化生成：输入一段文本，输出标准化 JSON 数组。

## 任务输出格式

模型输出必须是 JSON 数组。每个关系对象格式如下：

```json
[
  {
    "head_entity": "string",
    "tail_entity": "string",
    "relation_type": "ADE | DDI-MECHANISM | DDI-EFFECT | DDI-ADVISE | DDI-INT"
  }
]
```

如果文本中没有目标关系，输出必须是 `[]`。

仓库内部统一只使用下面五个标签：

- `ADE`
- `DDI-MECHANISM`
- `DDI-EFFECT`
- `DDI-ADVISE`
- `DDI-INT`

## 当前主线

如果你只想接手并继续当前推荐版本，先认这 5 个入口：

- 训练配置：`configs/qwen3_8b_lora_ddi_ade_final.yaml`
- 推理配置：`configs/infer_qwen3_8b_lora_ddi_ade_final.yaml`
- 训练数据：`data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`
- 验证数据：`data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- 测试数据：`data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_test.jsonl`

当前主线训练输出目录：

- `outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/`

当前最重要的一组基准评测归档：

- `results/benchmark_suite_vllm_batch64_20260327/`

## 仓库结构

### 顶层目录

- `configs/`
  训练和推理 YAML 配置。先看 `configs/README.md`。
- `data/`
  内部训练数据、增强样本和处理后数据。先看 `data/README.md`。
- `evaluate_datasets/`
  外部评测数据包、held-out 镜像和索引文件。先看 `evaluate_datasets/README.md`。
- `prompts/`
  训练与推理共用的系统提示词。先看 `prompts/README.md`。
- `scripts/`
  训练、推理、评估、分析、实验脚本入口。先看 `scripts/README.md`。
- `src/`
  核心 Python 模块与规则基线实现。先看 `src/README.md`。
- `resources/`
  规则基线使用的词典资源。先看 `resources/README.md`。
- `outputs/`
  训练产物、LoRA adapter、checkpoint 和可观测性文件。先看 `outputs/README.md`。
- `reports/`
  数据审计、训练摘要和统计报告。先看 `reports/README.md`。
- `results/`
  推理预测、指标文件和 benchmark 归档。先看 `results/README.md`。
- `flash_attn/`
  本地兼容层，用来补齐仓库需要的 `apply_rotary` 接口。先看 `flash_attn/README.md`。

### 建议阅读顺序

1. `README.md`
2. `configs/README.md`
3. `data/README.md`
4. `scripts/README.md`
5. `src/README.md`
6. `outputs/README.md`
7. `results/README.md`
8. `evaluate_datasets/README.md`

## 安装

```bash
git clone https://github.com/oceanusXXD/Comp6713-Ddi-Ade-Extraction.git
cd Comp6713-Ddi-Ade-Extraction
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 常用流程

### 1. 重新物化推荐数据

```bash
.venv/bin/python scripts/analysis/audit_and_prepare_final_dataset.py
```

这个命令会刷新：

- `data/processed/Comp6713-Ddi-Ade-Extraction_final/`
- `data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/`
- `reports/` 中的数据审计与统计文件

### 2. 做训练前 dry run

```bash
.venv/bin/python scripts/train/train_finetune.py \
  --config configs/qwen3_8b_lora_ddi_ade_final.yaml \
  --dry-run-samples 8
```

### 3. 训练当前主线 LoRA

```bash
.venv/bin/python scripts/train/train_finetune.py \
  --config configs/qwen3_8b_lora_ddi_ade_final.yaml \
  --do-train
```

### 4. 使用默认配置推理

```bash
.venv/bin/python scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_final.yaml
```

### 5. 评估预测结果

```bash
.venv/bin/python scripts/evaluation/evaluate_predictions.py \
  --predictions-path results/inference_runs/your_predictions.jsonl
```

### 6. 按增强类型拆分评估

```bash
.venv/bin/python scripts/evaluation/evaluate_predictions_by_augmentation.py \
  --predictions-path results/inference_runs/your_predictions.jsonl \
  --source-path data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train_augmentations.jsonl
```

## 命名约定


- 目录说明文件统一命名为 `README.md`
- 训练配置使用 `configs/<model>_<finetune>_<task>_<tag>.yaml`
- 推理配置使用 `configs/infer_<model>_<finetune>_<task>_<tag>.yaml`
- 处理后数据使用 `data/processed/<dataset_version>/`
- 训练输出使用 `outputs/<model>_<finetune>_<task>_<tag>/`
- 基准评测归档使用 `results/<suite>_<backend>_<batch>_<date>/`

现有历史目录名不会为了“好看”而强行重命名；判断是否是当前推荐版本，要优先看 `configs/` 中配置实际引用的路径。

## 文档索引

- `configs/README.md`
- `data/README.md`
- `scripts/README.md`
- `src/README.md`
- `src/baseline/README.md`
- `src/prevalidation/README.md`
- `evaluate_datasets/README.md`
- `outputs/README.md`
- `reports/README.md`
- `results/README.md`

## 备注

- `outputs/`、`results/` 里保留了较多历史实验产物；历史目录不等于当前主线。
- `evaluate_datasets/seen_style_core/` 是评测镜像，不是主线训练数据入口。
- 如果以后继续扩展仓库，建议优先保持“配置文件、目录 README、脚本入口”三者同步更新。
