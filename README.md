# COMP6713 DDI/ADE 抽取仓库说明

这个仓库用于医疗关系抽取任务，目标是从一段医疗文本中抽取两类关系：

- `ADE`：药物导致的不良反应
- `DDI-*`：药物和药物之间的相互作用

当前主线实现基于 `Qwen/Qwen3-8B` + LoRA，任务形式是结构化生成：输入一段文本，输出标准化 JSON 数组。

## 你应该先看哪里

如果你是第一次接手这个仓库，建议按下面顺序看：

1. `README.md`
   先了解仓库整体结构、当前主线配置和数据入口。
2. `configs/README.md`
   看训练配置和推理配置分别是什么。
3. `data/README.md`
   看数据文件、增强文件、处理后数据和主线实际使用的数据路径。
4. `scripts/README.md`
   看有哪些命令行入口脚本，以及训练 / 推理 / 评估分别怎么跑。
5. `src/README.md`
   看核心 Python 模块的职责分工。
6. `evaluate_datasets/README.md`
   看外部评测数据集和 `seen_style_core` 的定位。

## 任务定义

输入是一段医疗文本，输出必须是 JSON 数组。每个关系对象格式如下：

```json
[
  {
    "head_entity": "string",
    "tail_entity": "string",
    "relation_type": "ADE | DDI-MECHANISM | DDI-EFFECT | DDI-ADVISE | DDI-INT"
  }
]
```

如果文本中没有任何目标关系，输出必须是 `[]`。

仓库内部统一使用下面五个规范标签：

- `ADE`
- `DDI-MECHANISM`
- `DDI-EFFECT`
- `DDI-ADVISE`
- `DDI-INT`

## 当前主线是什么

当前主线训练和默认推理围绕下面两份配置展开：

- `configs/qwen3_8b_lora_ddi_ade_final.yaml`
  当前推荐训练配置。
- `configs/infer_qwen3_8b_lora_ddi_ade_final.yaml`
  当前推荐推理配置。

当前主线默认使用的数据是：

- 训练集：
  `data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`
- 验证集：
  `data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- 测试集：
  `data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_test.jsonl`

当前主线训练输出目录是：

- `outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/`

这个目录里目前主要是：

- `checkpoint-*`：训练中的阶段性 checkpoint
- `observability/`：训练过程记录、配置快照、参数统计、数据统计

## `seen_style_core` 

`evaluate_datasets/seen_style_core/` held-out 文件”整理进 `evaluate_datasets/` 体系，方便统一做评测管理。

它当前镜像的是：

- `data/merged_chatml_validation.jsonl`
- `data/merged_chatml_test.jsonl`

对应到评测目录里就是：

- `evaluate_datasets/seen_style_core/official_held_out/merged_chatml_validation.jsonl`
- `evaluate_datasets/seen_style_core/official_held_out/merged_chatml_test.jsonl`




## 顶层文件和目录说明

- `README.md`
  这个总说明文件。
- `requirements.txt`
  Python 依赖列表。
- `LICENSE`
  仓库许可证。
- `.gitignore`
  Git 忽略规则，主要排除本地模型、中间产物、缓存等。
- `.gitattributes`
  Git LFS 跟踪规则，主要用于较大的模型权重文件。
- `configs/`
  训练和推理 YAML 配置文件。详见 `configs/README.md`。
- `data/`
  内部训练 / 验证 / 测试数据，以及清洗增强后的处理结果。详见 `data/README.md`。
- `evaluate_datasets/`
  外部评测数据包和同风格 held-out 镜像。详见 `evaluate_datasets/README.md`。
- `prompts/`
  训练和推理共用的系统提示词。详见 `prompts/README.md`。
- `scripts/`
  训练、推理、评估、分析、实验脚本入口。详见 `scripts/README.md`。
- `src/`
  核心 Python 模块。详见 `src/README.md`。
- `resources/`
  基线系统使用的词典资源。详见 `resources/README.md`。
- `outputs/`
  训练输出、LoRA 适配器、可观测性产物和基线输出。详见 `outputs/README.md`。
- `reports/`
  数据审计和训练摘要报告。详见 `reports/README.md`。
- `results/`
  推理预测、指标文件和变体实验结果。详见 `results/README.md`。
- `flash_attn/`
  一个本地兼容层，用来提供仓库需要的 `apply_rotary` 接口。详见 `flash_attn/README.md`。

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

### 1. 重新物化推荐训练数据

```bash
.venv/bin/python scripts/analysis/audit_and_prepare_final_dataset.py
```

这个命令会更新：

- `data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/`
- `reports/` 里的相关审计报告

### 2. 训练当前主线 LoRA

先做 dry run：

```bash
.venv/bin/python scripts/train/train_finetune.py \
  --config configs/qwen3_8b_lora_ddi_ade_final.yaml \
  --dry-run-samples 8
```

正式训练：

```bash
.venv/bin/python scripts/train/train_finetune.py \
  --config configs/qwen3_8b_lora_ddi_ade_final.yaml \
  --do-train
```

### 3. 使用默认配置推理

```bash
.venv/bin/python scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_final.yaml
```

### 4. 评估预测结果

```bash
.venv/bin/python scripts/evaluation/evaluate_predictions.py \
  --predictions-path results/your_predictions.jsonl
```

### 5. 按增强类型拆分评估

```bash
.venv/bin/python scripts/evaluation/evaluate_predictions_by_augmentation.py \
  --predictions-path results/your_predictions.jsonl \
  --source-path data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train_augmentations.jsonl
```

## README 导航

- `configs/README.md`
  看配置文件具体字段和用途。
- `data/README.md`
  看训练集、增强集、处理后数据和 manifest。
- `evaluate_datasets/README.md`
  看外部评测包、索引文件和下载脚本。
- `scripts/README.md`
  看所有命令行入口及推荐命令。
- `src/README.md`
  看核心模块职责分工。
- `src/baseline/README_baseline.md`
  看规则基线的流程和脚本。
- `src/prevalidation/README_pretest.md`
  看旧版预验证脚本和零样本测试工具。
- `outputs/README.md`
  看输出目录里各类产物的含义。
- `outputs/qwen3_8b_lora/final_adapter/README.md`
  看旧版已保留 adapter 目录里每个文件的含义。
- `reports/README.md`
  看审计和训练摘要文件。
- `results/README.md`
  看结果文件命名规范和历史实验产物。

## 备注

- `outputs/`、`results/` 和部分 `evaluate_datasets/` 内容里存在历史实验产物，名字不一定都代表“当前推荐版本”。
- 判断当前主线应该优先看 `configs/` 里的最新配置，再看它引用的数据路径和输出路径。
- 如果后面继续整理仓库，建议优先保持“配置文件、README 和实际脚本入口”三者一致。
