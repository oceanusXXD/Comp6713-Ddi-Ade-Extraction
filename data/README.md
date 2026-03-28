# 数据目录说明

这个目录保存仓库内部数据资产。整理后，`data/` 顶层只保留核心 split，增强相关文件和 guardrail 小样本分别收进了独立子目录。

## 当前主线先看哪里

如果你只想跑当前推荐版本，先看：

- `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`
- `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_test.jsonl`
- `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train_augmentations.jsonl`

当前数据规模如下：

- 训练集：`4924`
- 验证集：`436`
- 测试集：`619`
- 增强 sidecar：`1012`

## 顶层现在保留什么

### 核心基础 split

- `merged_chatml_train.jsonl`
- `merged_chatml_validation.jsonl`
- `merged_chatml_test.jsonl`

这三份基础 ChatML 文件仍然保留在顶层，原因是它们仍被若干基线脚本、预验证脚本和 `seen_style_core` 镜像直接引用。

## 子目录

### `augmentations/`

增强规格和历史增强数据统一收进这里：

- `augmentations/curated_train_augmentations.json`
- `augmentations/curated_train_augmentations_supplement_augment.json`
- `augmentations/merged_chatml_train_augmentations.jsonl`
- `augmentations/merged_chatml_train_augmented.jsonl`

### `guardrails/`

- `guardrails/general_chatml_test.jsonl`
  偏离医学抽取任务的通用负样本 / 守护样本集合。

### `processed/`

处理后数据版本统一放在这里。

#### `processed/Comp6713-Ddi-Ade-Extraction_final/`

这是一版清洗后的基础数据，不包含当前主线推荐增强混入训练集。

- `merged_chatml_train.jsonl`
- `merged_chatml_validation.jsonl`
- `merged_chatml_test.jsonl`
- `merged_chatml_train_augmentations.jsonl`
- `manifest.json`

#### `processed/Comp6713-Ddi-Ade-Extraction_final_augment/`

这是当前主线实际使用的数据版本。

- `merged_chatml_train.jsonl`
- `merged_chatml_validation.jsonl`
- `merged_chatml_test.jsonl`
- `merged_chatml_train_augmentations.jsonl`
- `manifest.json`

## 数据格式

主训练 / 推理文件使用 ChatML JSONL。每行至少包含：

- `messages`
  对话消息数组，通常包括 `system`、`user`、`assistant`。

增强 sidecar 额外包含：

- `augmentation_type`
  增强类型，例如 `paraphrase`、`negative`、`hardcase`、`margincase`。

## 标签规范

仓库内部只使用下面五个规范标签：

- `ADE`
- `DDI-MECHANISM`
- `DDI-EFFECT`
- `DDI-ADVISE`
- `DDI-INT`

历史文件中可能出现大小写变体；评估逻辑会做规范化，但新数据和新输出建议始终使用全大写规范标签。

## 谁会生成或消费这些文件

- `scripts/analysis/audit_and_prepare_final_dataset.py`
  读取 `augmentations/` 下的增强规格与 sidecar，并物化 `processed/...`
- `scripts/train/train_finetune.py`
  消费 `processed/...` 下的训练 / 验证集
- `scripts/inference/predict.py`
  默认消费 `processed/...final_augment/...` 下的验证集
- `scripts/evaluation/evaluate_predictions_by_augmentation.py`
  消费处理后增强 sidecar 做分组评估

## 使用建议

- 跑当前主线训练：用 `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`
- 跑当前默认验证 / 推理：用 `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- 回看基础不增强版本：看 `processed/Comp6713-Ddi-Ade-Extraction_final/`
- 改增强规格：去 `augmentations/`
- 看守护样本：去 `guardrails/`

## 注意事项

- 顶层 `merged_chatml_validation.jsonl` 与 `merged_chatml_test.jsonl` 仍然保留，但不是当前主线默认入口。
- `evaluate_datasets/seen_style_core/` 镜像的是顶层基础验证 / 测试文件，不是 `processed/...final_augment/...`。
- 如果只是想复现实验主线，不要直接把历史 `augmentations/merged_chatml_train_augmented.jsonl` 当作默认训练入口。
