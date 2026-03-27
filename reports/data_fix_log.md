# 数据修复日志

## 输入

- 上一版增强 sidecar：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/merged_chatml_train_augmentations.jsonl`
- 主增强规格文件：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/curated_train_augmentations.json`
- 补充增强规格文件：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/curated_train_augmentations_supplement_augment.json`

说明：当前仓库已统一保留 `r4` 补充规格文件；基础版 `final` 的这 16 条新增样本可视为该文件中的历史子集。

## 处理动作

- 移除格式损坏行数：0
- 移除非法标签行数：0
- 移除实体边界错误行数：0
- 合并时移除的重复增强规格数：`0`
- 从处理后训练集中移除的精确重叠原始训练样本：`5`
- 保留的历史有效增强样本：`48`
- 新增人工整理增强样本：`16`

## 补充重点

- 增加了更多 `DDI-INT`，覆盖类级别和实例级别的相互作用表述。
- 增加了更多 `DDI-ADVISE` / `DDI-MECHANISM`，以改善标签平衡。
- 增加了更多高迷惑性的 `negative` 与 `margincase` 样本，以强化决策边界。
- 保持所有可训练文件与 ChatML JSONL 格式兼容。

## 最终产物

- 最终训练集：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final/merged_chatml_train.jsonl`
- 最终验证集：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final/merged_chatml_validation.jsonl`
- 最终测试集：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final/merged_chatml_test.jsonl`
- 最终增强 sidecar：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final/merged_chatml_train_augmentations.jsonl`

## 最终规模

- 原始训练集行数：`3917`
- 去除划分污染后的基础训练集行数：`3912`
- 最终增强集行数：`64`
- 最终合并训练集行数：`3976`
- 最终数据集哈希：`c5fc8c06`
