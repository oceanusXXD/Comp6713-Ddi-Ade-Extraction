# 数据审计报告

## 范围

- 仓库：`data/Comp6713-Ddi-Ade-Extraction`
- 最终处理后数据集：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final`
- 本次审计对应的数据集哈希：`c5fc8c06`

## 检查文件

- 原始训练集：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/merged_chatml_train.jsonl`
- 原始验证集：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/merged_chatml_validation.jsonl`
- 原始测试集：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/merged_chatml_test.jsonl`
- 现有增强 sidecar：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/merged_chatml_train_augmentations.jsonl`
- 最终增强 sidecar：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final/merged_chatml_train_augmentations.jsonl`
- 最终合并训练集：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final/merged_chatml_train.jsonl`

## 格式一致性

- 训练数据统一采用 ChatML JSONL 格式，结构为 `messages = [system, user, assistant]`。
- `span / offset / index` 字段不属于本仓库 schema，因此不适用边界偏移检查。
- 原始训练集解析失败数：`0`
- 最终增强集解析失败数：`0`
- 最终训练集解析失败数：`0`
- 最终增强集实体子串问题数：`0`
- 最终增强集非法标签数：`0`

## 语义与标签质量

- 从上一版本保留的已整理增强样本：`48`
- 审计后新增补充样本：`16`
- 合并过程中去掉的重复增强规格数：`0`
- 最终增强集中文本完全重复的样本数：`0`
- 最终增强集中近重复样本对数量（token Jaccard >= 0.75）：`0`
- 增强集与验证集的精确重叠：`0` 行
- 增强集与测试集的精确重叠：`0` 行
- 从处理后训练集中剔除的污染原始训练样本：`5`

人工复核结论：
- `paraphrase`：保持语义一致，同时变化句法和话语表达形式。
- `negative`：保留药物、事件、相互作用等表面线索，但刻意不提供成立关系所需证据。
- `hardcase`：包含干扰实体、类与实例切换、并列结构或长距离触发因素。
- `margincase`：贴近正例证据，但由于不确定性、混杂因素或时序歧义，仍不应判为确定关系。

## 分布与覆盖

- 原始训练集行数：`3917`
- 最终增强集行数：`64`
- 最终合并训练集行数：`3976`
- 最终增强类型分布：`{"hardcase": 16, "margincase": 16, "negative": 16, "paraphrase": 16}`
- 最终增强关系分布：`{"ADE": 16, "DDI-ADVISE": 9, "DDI-EFFECT": 11, "DDI-INT": 13, "DDI-MECHANISM": 9}`
- 最终增强集中正样本行数：`32`
- 最终增强集中空目标行数：`32`
- 最终训练集中空目标行数：`32`
- 最终训练集标签分布：`{"ADE": 5385, "DDI-ADVISE": 749, "DDI-EFFECT": 1486, "DDI-INT": 195, "DDI-MECHANISM": 1161}`

覆盖亮点：
- 最终增强集中的长距离关系数：`38`
- 最终增强集中的复杂词汇标记统计：`{"conditional": 6, "coordination": 58, "negation": 27, "speculative": 5}`
- 各增强分组统计摘要保存在 `reports/data_stats.json`。

## 划分污染检查

- 原始训练集与验证集精确重叠：`2` 行
- 原始训练集与测试集精确重叠：`2` 行
- 验证集与测试集精确重叠：`0` 行
- 最终训练集与验证集精确重叠：`0` 行
- 最终训练集与测试集精确重叠：`0` 行
- 最终增强集与验证集精确重叠：`0` 行
- 最终增强集与测试集精确重叠：`0` 行

## 关键结论

- 最终物化后，仓库内数据 schema 保持内部一致且可解析。
- 原始数据集中没有空目标样本，因此 `negative` 与 `margincase` 填补了真实训练缺口。
- 稀有 DDI 子类，尤其是 `DDI-INT` 和 `DDI-ADVISE`，在首轮增强中相对稀疏，因此做了定向补充。
- 原始划分中存在 train/dev 与 train/test 的精确重叠，这些样本已经从处理后的训练集中移除。
