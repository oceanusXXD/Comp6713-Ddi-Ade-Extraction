# COMP6713 DDI/ADE Extraction

用于医疗关系抽取任务的训练、推理、评估与数据分析代码仓库。当前实现基于 `Qwen/Qwen3-8B` 和 LoRA，任务形式为结构化生成：输入一段医疗文本，输出标准化的 ADE/DDI 关系 JSON。

## 功能概览

- LoRA 微调
- 批量推理
- 结果评估
- 数据集统计分析

## 任务定义

输入是一段医疗文本，输出是一个 JSON 数组，每个元素格式如下：

```json
[
  {
    "head_entity": "string",
    "tail_entity": "string",
    "relation_type": "ADE | DDI-MECHANISM | DDI-EFFECT | DDI-ADVISE | DDI-INT"
  }
]
```

如果没有关系，输出必须为 `[]`。

支持的关系标签：

- `ADE`
- `DDI-MECHANISM`
- `DDI-EFFECT`
- `DDI-ADVISE`
- `DDI-INT`

## 项目结构

```text
Comp6713-Ddi-Ade-Extraction/
├── README.md
├── requirements.txt
├── configs/
│   ├── infer_qwen3_8b.yaml
│   └── train_qwen3_8b_lora.yaml
├── data/
│   ├── README.md
│   ├── merged_chatml_train.jsonl
│   ├── merged_chatml_validation.jsonl
│   └── merged_chatml_test.jsonl
├── outputs/
│   └── qwen3_8b_lora/
│       └── final_adapter/
├── scripts/
│   ├── analysis/
│   │   └── analyze_dataset.py
│   ├── evaluation/
│   │   └── evaluate_predictions.py
│   ├── inference/
│   │   └── predict.py
│   └── train/
│       └── train_finetune.py
└── src/
```

说明：

- `data/README.md` 说明数据格式与标签约定。
- `outputs/qwen3_8b_lora/final_adapter/` 保存最终 LoRA 适配器。
- `models/`、`results/`、checkpoint 和其他中间输出默认不纳入版本控制。

## 安装

```bash
git clone https://github.com/oceanusXXD/Comp6713-Ddi-Ade-Extraction.git
cd Comp6713-Ddi-Ade-Extraction
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 数据

仓库当前使用 ChatML JSONL 数据：

- `data/merged_chatml_train.jsonl`
- `data/merged_chatml_validation.jsonl`
- `data/merged_chatml_test.jsonl`

更详细的数据格式说明见 [`data/README.md`](data/README.md)。

## 模型准备

默认 base model 使用 Hugging Face 模型 ID：

```text
Qwen/Qwen3-8B
```

如果需要下载到本地后再运行：

```bash
huggingface-cli login
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/base/Qwen3-8B
```

下载后可将配置中的：

```yaml
model:
  base_model_name_or_path: Qwen/Qwen3-8B
```

替换为：

```yaml
model:
  base_model_name_or_path: ./models/base/Qwen3-8B
```

最终 LoRA 适配器位于：

```text
outputs/qwen3_8b_lora/final_adapter/
```

## 配置文件

训练配置：

- `configs/train_qwen3_8b_lora.yaml`

推理配置：

- `configs/infer_qwen3_8b.yaml`

推理配置中的关键字段：

```yaml
model:
  base_model_name_or_path: Qwen/Qwen3-8B
  adapter_path: ./outputs/qwen3_8b_lora/final_adapter

data:
  input_path: data/merged_chatml_validation.jsonl
```

说明：

- `base_model_name_or_path` 可以是 Hugging Face 模型 ID 或本地目录。
- `adapter_path` 设为 `null` 时表示只使用 base model 推理。
- `input_path` 可以替换为自定义数据文件。

## 训练

先运行 dry run 检查配置、tokenizer 和样本编码：

```bash
.venv/bin/python scripts/train/train_finetune.py \
  --config configs/train_qwen3_8b_lora.yaml \
  --dry-run-samples 8
```

正式训练：

```bash
.venv/bin/python scripts/train/train_finetune.py \
  --config configs/train_qwen3_8b_lora.yaml \
  --do-train
```

从 checkpoint 恢复训练：

```bash
.venv/bin/python scripts/train/train_finetune.py \
  --config configs/train_qwen3_8b_lora.yaml \
  --do-train \
  --resume-from-checkpoint outputs/qwen3_8b_lora/checkpoint-100
```

## 推理

使用默认验证集推理：

```bash
.venv/bin/python scripts/inference/predict.py \
  --config configs/infer_qwen3_8b.yaml
```

使用自定义数据集推理：

```bash
.venv/bin/python scripts/inference/predict.py \
  --config configs/infer_qwen3_8b.yaml \
  --input-path data/your_dataset.jsonl \
  --output-path results/your_dataset_predictions.jsonl \
  --metrics-path results/your_dataset_metrics.txt \
  --metrics-json-path results/your_dataset_metrics.json
```

单条文本推理：

```bash
.venv/bin/python scripts/inference/predict.py \
  --config configs/infer_qwen3_8b.yaml \
  --input-text "Erythromycin may increase the serum concentration of simvastatin."
```

默认推理 backend 为 `vllm`，也支持 `transformers`。

## 评估

评估预测文件：

```bash
.venv/bin/python scripts/evaluation/evaluate_predictions.py \
  --predictions-path results/your_dataset_predictions.jsonl
```

如果预测文件中不包含 gold，可额外提供 gold 数据集：

```bash
.venv/bin/python scripts/evaluation/evaluate_predictions.py \
  --predictions-path results/your_dataset_predictions.jsonl \
  --gold-path data/merged_chatml_test.jsonl \
  --split test
```

## 数据分析

统计数据集长度与 token 分布：

```bash
.venv/bin/python scripts/analysis/analyze_dataset.py \
  --config configs/train_qwen3_8b_lora.yaml \
  --input-path data/merged_chatml_train.jsonl \
  --output-path outputs/qwen3_8b_lora/observability/manual_train_dataset_stats.json
```

## 版本控制说明

- `models/` 默认忽略，用于本地保存 base model。
- `results/`、checkpoint、observability 和其他中间产物默认忽略。
- `outputs/qwen3_8b_lora/final_adapter/` 是需要保留的最终 LoRA 目录。
- `outputs/qwen3_8b_lora/final_adapter/*.safetensors` 已配置为 Git LFS 跟踪。

首次在本机使用前建议执行：

```bash
git lfs install
```

## License

见 [`LICENSE`](LICENSE)。
