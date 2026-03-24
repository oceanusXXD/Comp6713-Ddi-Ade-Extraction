# Comp6713-Ddi-Ade-Extraction
COMP6713 team project on DDI and ADE extraction with model training, inference, evaluation, and demo.

## 数据合并策略 (Data Merging Strategy)
- **训练集 (Merged Train)**: `ade_unified_train.jsonl` + 90% `ddi_unified_train.jsonl`。合并后进行全局打乱 (Global Shuffle)。
- **验证集 (Mixed Validation)**: `ade_unified_validation.jsonl` + 10% `ddi_unified_train.jsonl`。此策略既保留了 ADE 原有的验证价值，又为 DDI 任务安插了数据"监军"，能同时监控两个任务的训练状态。
- **测试集 (Merged Test)**: `ade_unified_test.jsonl` + `ddi_unified_test.jsonl`。这是最后的考卷，绝对不参与训练。
