# 评测数据目录说明

这个目录统一管理仓库里的评测数据资产，包括：

- 外部公开评测数据包
- 仓库内部 held-out 文件的评测镜像
- 数据清单、索引和下载脚本

## 当前目录中的关键文件

- `README.md`
  目录总说明。
- `download_evaluate_datasets.sh`
  下载与整理入口脚本。
- `build_dataset_index.py`
  重建评测索引。
- `MANIFEST.json`
  下载 / 整理脚本产出的 bundle 清单。
- `DATASET_INDEX.json`
  机器可读的评测数据索引。
- `DATASET_INDEX.md`
  人类可读的评测数据索引。

## 子目录

- `seen_style_core/`
  同风格 held-out 评测镜像，不是新数据集。
- `ddi_transfer/`
  DDI 迁移评测包。
- `ade_transfer/`
  ADE 迁移评测包。
- `pharmacovigilance_cross_genre/`
  药物警戒跨体裁评测包。
- `general_guardrails/`
  通用守护、长上下文和泛化评测包。

## `seen_style_core` 的定位

需要明确区分：

- 当前主线训练：不用 `seen_style_core`
- 当前默认验证 / 默认推理：也不直接用 `seen_style_core`
- 额外做同风格补充评测：可以用 `seen_style_core`

它当前镜像的是：

- `../data/merged_chatml_validation.jsonl`
- `../data/merged_chatml_test.jsonl`

在这个目录里对应为：

- `seen_style_core/official_held_out/merged_chatml_validation.jsonl`
- `seen_style_core/official_held_out/merged_chatml_test.jsonl`

所以它的本质是“评测入口整理”，不是“主线训练数据入口”。

## 当前主线真正使用的数据

主线训练 / 默认验证 / 默认测试用的仍然是：

- `../data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`
- `../data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- `../data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_test.jsonl`

## 常用命令

### 下载并整理评测数据

```bash
bash evaluate_datasets/download_evaluate_datasets.sh
```

强制重下：

```bash
bash evaluate_datasets/download_evaluate_datasets.sh --force
```

### 重建索引

```bash
.venv/bin/python evaluate_datasets/build_dataset_index.py
```

## 脚本职责

- `download_evaluate_datasets.sh`
  轻量 shell 包装器。
- `scripts/analysis/fetch_evaluate_datasets.py`
  下载公开数据、解压压缩包、写 `MANIFEST.json`，并构建 `seen_style_core/` 镜像。
- `build_dataset_index.py`
  读取 `MANIFEST.json` 与内部数据路径，生成 `DATASET_INDEX.json` 和 `DATASET_INDEX.md`。

## 使用建议

- 想做同风格补充评测：看 `seen_style_core/`
- 想做 DDI 迁移评测：看 `ddi_transfer/`
- 想做 ADE 迁移评测：看 `ade_transfer/`
- 想做跨体裁药物警戒评测：看 `pharmacovigilance_cross_genre/`
- 想做长上下文 / 守护型评测：看 `general_guardrails/`
