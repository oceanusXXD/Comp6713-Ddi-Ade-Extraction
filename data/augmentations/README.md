# 增强数据目录说明

这个目录集中保存增强规格和增强相关的历史数据文件。

## 当前内容

- `curated_train_augmentations.json`
  主增强规格文件。
- `curated_train_augmentations_supplement_augment.json`
  对主增强规格的补充集合。
- `merged_chatml_train_augmentations.jsonl`
  已转成 ChatML JSONL 的增强 sidecar。
- `merged_chatml_train_augmented.jsonl`
  历史上的“训练集 + 增强样本”合并版本。

## 说明

- 当前主线训练不直接读取这里的历史合并文件。
- `scripts/analysis/audit_and_prepare_final_dataset.py` 会读取这里的规格与 sidecar，再物化到 `data/processed/`。
