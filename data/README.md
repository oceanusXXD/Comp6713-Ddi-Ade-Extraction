# 数据目录说明

这个目录保存仓库内部数据资产，既包括较早期的基础 ChatML 划分，也包括当前主线真正使用的处理后数据。

## 先记住主线数据

当前主线训练 / 默认验证 / 默认测试，实际使用的是：

- `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`
- `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_test.jsonl`

如果你只是跑现在的主线配置，优先看这三个文件。

## 当前文件说明

### 顶层基础文件

- `merged_chatml_train.jsonl`
  仓库保留的基础训练集 ChatML 文件。
- `merged_chatml_validation.jsonl`
  仓库保留的基础验证集 ChatML 文件；也是 `seen_style_core` 镜像来源之一。
- `merged_chatml_test.jsonl`
  仓库保留的基础测试集 ChatML 文件；也是 `seen_style_core` 镜像来源之一。
- `merged_chatml_train_augmentations.jsonl`
  已转成 ChatML JSONL 的增强样本 sidecar，每行带 `augmentation_type`。
- `merged_chatml_train_augmented.jsonl`
  历史上的“训练集 + 增强样本”合并版本，主要用于回看早期实验。
- `curated_train_augmentations.json`
  结构化增强样本源文件，保存增强类型、原始文本和目标关系。
- `curated_train_augmentations_supplement_augment.json`
  对增强样本的补充集合，用于扩展 `curated_train_augmentations.json`。
- `general_chatml_test.jsonl`
  通用负样本 / 守护样本小集合，内容故意偏离医学抽取任务，通常用于检查模型是否会在非医学文本上乱抽关系。

### 处理后数据目录

#### `processed/Comp6713-Ddi-Ade-Extraction_final/`

- `merged_chatml_train.jsonl`
  清洗后的基础训练集。
- `merged_chatml_validation.jsonl`
  清洗后的基础验证集。
- `merged_chatml_test.jsonl`
  清洗后的基础测试集。
- `merged_chatml_train_augmentations.jsonl`
  与这套基础划分对应的增强样本 sidecar。
- `manifest.json`
  这套数据的清单文件，记录数据哈希、路径和构建信息。

#### `processed/Comp6713-Ddi-Ade-Extraction_final_augment/`

- `merged_chatml_train.jsonl`
  当前主线训练集，已融合推荐增强样本。
- `merged_chatml_validation.jsonl`
  当前主线验证集。
- `merged_chatml_test.jsonl`
  当前主线测试集。
- `merged_chatml_train_augmentations.jsonl`
  当前主线增强样本 sidecar，可用于按 `augmentation_type` 做切分评估。
- `manifest.json`
  当前主线处理后数据的清单文件，记录哈希、路径和构建信息。

## 数据格式

大多数训练 / 推理输入文件都采用 ChatML JSONL。每一行是一个 JSON 对象，至少包含：

- `messages`
  对话消息数组，通常包括 `system`、`user`、`assistant` 三类角色。

增强样本 sidecar 额外会包含：

- `augmentation_type`
  增强类型，例如 `paraphrase`、`negative`、`hardcase`、`margincase`。

示例：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert medical information extraction system..."
    },
    {
      "role": "user",
      "content": "Erythromycin may increase the serum concentration of simvastatin."
    },
    {
      "role": "assistant",
      "content": "[{\"head_entity\": \"Erythromycin\", \"tail_entity\": \"simvastatin\", \"relation_type\": \"DDI-MECHANISM\"}]"
    }
  ]
}
```

## 标签规范

仓库内部规范标签只有五个：

- `ADE`
- `DDI-MECHANISM`
- `DDI-EFFECT`
- `DDI-ADVISE`
- `DDI-INT`

历史文件中可能出现 `DDI-mechanism` 这类大小写变体，评估和解析逻辑会做规范化，但新输出建议一律用上面的全大写形式。

## 这些数据分别什么时候用

- 跑当前主线训练：
  使用 `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`
- 跑当前默认验证 / 推理：
  使用 `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- 跑当前默认测试评估：
  使用 `processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_test.jsonl`
- 分析增强样本质量：
  看 `merged_chatml_train_augmentations.jsonl` 或 `processed/.../merged_chatml_train_augmentations.jsonl`
- 回看基础不增强版本：
  看 `processed/Comp6713-Ddi-Ade-Extraction_final/`
- 做同风格补充评测：
  不在这里直接用，去看 `evaluate_datasets/seen_style_core/`

## 谁会生成或消费这些文件

- `scripts/analysis/audit_and_prepare_final_dataset.py`
  负责物化 `processed/...` 下的推荐数据和 manifest。
- `scripts/train/train_finetune.py`
  消费训练集 / 验证集路径。
- `scripts/inference/predict.py`
  消费推理输入路径。
- `scripts/evaluation/evaluate_predictions_by_augmentation.py`
  会读取带 `augmentation_type` 的增强 sidecar 来做分组评估。

## 注意事项

- `merged_chatml_validation.jsonl` 与 `merged_chatml_test.jsonl` 虽然仍保留，但不是当前主线默认验证 / 默认测试入口。
- `evaluate_datasets/seen_style_core/` 镜像的是顶层基础 `merged_chatml_validation.jsonl` 和 `merged_chatml_test.jsonl`，不是 `processed/...final_augment/...`。
- 如果你只想跑“现在推荐的版本”，尽量不要直接拿历史 `merged_chatml_train_augmented.jsonl` 作为训练入口。
