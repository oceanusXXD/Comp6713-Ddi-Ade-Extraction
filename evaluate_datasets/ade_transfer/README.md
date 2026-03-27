# ADE 迁移评测包说明

这个目录收集和 ADE 迁移评测相关的公开数据。

## 目录里的内容

### `ADE_Corpus_V2/`

- `ADE_Corpus_V2/raw/Ade_corpus_v2_drug_ade_relation.parquet`
  从公开 Hugging Face 数据源下载的原始 parquet 文件。
- `ADE_Corpus_V2/processed/drug_ade_relation.jsonl`
  从 parquet 导出的 JSONL 文件，方便后续脚本读取和分析。

### `PHEE/`

- `PHEE/raw/train.json`
  官方训练划分。
- `PHEE/raw/dev.json`
  官方开发划分。
- `PHEE/raw/test.json`
  官方测试划分。

## 这个包适合什么时候用

- 你想看模型对药物不良事件相关任务的迁移能力
- 你想把仓库当前任务和公开 ADE 数据做横向比较

## 说明

- `ADE_Corpus_V2` 这里保留了“原始下载文件 + 处理后 JSONL”两种形态。
- `PHEE` 这里保留的是仓库当前明确使用的公开划分文件，不代表后续版本的全部资源。
