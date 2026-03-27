# 通用守护评测包说明

这个目录收集不完全属于 DDI / ADE 主任务，但对模型泛化、长上下文能力和指令约束能力有帮助的评测数据。

## 当前子目录说明

- `IFEval/`
  指令遵循相关评测数据。当前保留 `raw/ifeval_input_data.jsonl`。
- `DocRED/`
  文档级关系抽取相关数据。当前保留多个 `.json.gz` 原始文件。
- `LongBench/`
  长上下文评测数据，其中 `LongBench/light/` 是仓库裁出来的轻量子集。

## 当前文件和子目录各自干什么

### `IFEval/`

- `IFEval/raw/ifeval_input_data.jsonl`
  指令遵循评测输入。

### `DocRED/`

- `DocRED/raw/train_annotated.json.gz`
  训练集原始压缩文件。
- `DocRED/raw/dev.json.gz`
  开发集原始压缩文件。
- `DocRED/raw/test.json.gz`
  测试集原始压缩文件。
- `DocRED/raw/rel_info.json.gz`
  关系标签说明。

### `LongBench/`

- `LongBench/raw/data.zip`
  LongBench 原始压缩包。
- `LongBench/light/`
  从完整数据里裁出来的轻量子集，详见 `LongBench/light/README.md`。

## 适合什么时候用

- 想检查模型是否只会做主任务、还是具备一定泛化能力
- 想做长上下文或守护型回归测试
- 想在不依赖医疗语料的情况下看模型输出是否仍然稳定
