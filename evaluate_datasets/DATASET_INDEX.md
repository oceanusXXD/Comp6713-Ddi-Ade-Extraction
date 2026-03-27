# 评测数据索引

## 当前内部训练 / 验证 / 测试数据

- 训练数据：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`，共 `4924` 条
- 验证数据：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`，共 `436` 条
- 测试数据：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_test.jsonl`，共 `619` 条
- 增强 sidecar：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train_augmentations.jsonl`，共 `1012` 条
- 增强类型分布：`{"hardcase": 247, "margincase": 265, "negative": 320, "paraphrase": 180}`

## 当前外部评测数据

- 同风格 held-out 验证集：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/evaluate_datasets/seen_style_core/official_held_out/merged_chatml_validation.jsonl`，共 `436` 条
- 同风格 held-out 测试集：`/home/coder/data/Comp6713-Ddi-Ade-Extraction/evaluate_datasets/seen_style_core/official_held_out/merged_chatml_test.jsonl`，共 `619` 条

## 外部 bundle 目录

- `ade_transfer`：状态 `complete`
- `ddi_transfer`：状态 `complete`
- `general_guardrails`：状态 `complete`
- `pharmacovigilance_cross_genre`：状态 `complete`
- `seen_style_core`：状态 `partial`

## 用法

- 重新下载并整理外部评测集：`bash evaluate_datasets/download_evaluate_datasets.sh`
- 重新生成索引：`.venv/bin/python evaluate_datasets/build_dataset_index.py`
