# 评测数据目录说明

这个目录用于统一管理仓库里的评测数据资产，包含三类东西：

- 外部公开评测数据包
- 仓库内部同风格 held-out 文件的评测镜像
- 数据索引、下载脚本和 manifest

## 当前目录里的文件是干什么的

- `README.md`
  这个总说明文件。
- `download_evaluate_datasets.sh`
  Shell 入口，用来调用底层 Python 脚本下载和整理评测数据。
- `build_dataset_index.py`
  根据当前目录内容和内部数据文件生成索引文件。
- `MANIFEST.json`
  下载 / 整理脚本写出的清单文件，记录各 bundle 的来源和状态。
- `DATASET_INDEX.json`
  机器可读的评测数据索引。
- `DATASET_INDEX.md`
  人类可读的评测数据索引说明。

## 子目录说明

- `seen_style_core/`
  同风格 held-out 评测镜像，不是新数据集。详见 `seen_style_core/README.md`。
- `ddi_transfer/`
  DDI 迁移评测包，包含公开的 `DDIExtraction 2013` 和 `TAC 2018 DDI`。详见 `ddi_transfer/README.md`。
- `ade_transfer/`
  ADE 迁移评测包，包含 `ADE-Corpus-V2` 和 `PHEE`。详见 `ade_transfer/README.md`。
- `pharmacovigilance_cross_genre/`
  药物警戒跨体裁评测包，包含 `TAC 2017 ADR` 和 `CADEC v2`。详见 `pharmacovigilance_cross_genre/README.md`。
- `general_guardrails/`
  通用约束、长上下文和泛化守护评测包。详见 `general_guardrails/README.md`。

## `seen_style_core` 到底算“用了还是没用”

明确一点：

- 当前主线训练：不用 `seen_style_core`
- 当前默认验证 / 默认推理：也不直接用 `seen_style_core`
- 做额外同风格评测：可以用 `seen_style_core`

它当前镜像的是：

- `../data/merged_chatml_validation.jsonl`
- `../data/merged_chatml_test.jsonl`

在这个目录里对应为：

- `seen_style_core/official_held_out/merged_chatml_validation.jsonl`
- `seen_style_core/official_held_out/merged_chatml_test.jsonl`

所以它的本质是“评测入口整理”，不是“主线训练数据入口”。

## 当前主线真正使用的数据

主线训练 / 默认验证 / 默认测试用的仍然是内部处理后数据：

- `../data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`
- `../data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- `../data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_test.jsonl`

## 常用命令

### 下载并整理评测数据

```bash
bash evaluate_datasets/download_evaluate_datasets.sh
```

如果需要强制重下：

```bash
bash evaluate_datasets/download_evaluate_datasets.sh --force
```

### 重建索引

```bash
.venv/bin/python evaluate_datasets/build_dataset_index.py
```

## 脚本各自做什么

- `download_evaluate_datasets.sh`
  只是一个轻量包装器，负责调用 `scripts/analysis/fetch_evaluate_datasets.py`。
- `scripts/analysis/fetch_evaluate_datasets.py`
  负责下载公开数据、解压压缩包、生成 `MANIFEST.json`，并把内部 held-out 文件镜像成 `seen_style_core/`。
- `build_dataset_index.py`
  读取 `MANIFEST.json` 和当前内部数据路径，统计行数，生成 `DATASET_INDEX.json` 和 `DATASET_INDEX.md`。

## 使用建议

- 想做同风格补充评测：看 `seen_style_core/`
- 想做 DDI 迁移评测：看 `ddi_transfer/`
- 想做 ADE 迁移评测：看 `ade_transfer/`
- 想做跨体裁药物警戒评测：看 `pharmacovigilance_cross_genre/`
- 想做长上下文 / 守护型评测：看 `general_guardrails/`
