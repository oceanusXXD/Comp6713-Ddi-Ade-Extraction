# LongBench 轻量子集说明

这个目录是从 `LongBench` 原始数据里裁出来的轻量子集，用来做更快的长上下文守护测试。

## 当前文件说明

- `multifieldqa_en.jsonl`
  英文多字段问答子集。
- `multifieldqa_zh.jsonl`
  中文多字段问答子集。
- `passage_retrieval_en.jsonl`
  英文段落检索子集。
- `passage_retrieval_zh.jsonl`
  中文段落检索子集。
- `gov_report.jsonl`
  政府报告类长文摘要 / 理解子集。
- `vcsum.jsonl`
  中文长文本摘要子集。
- `README.md`
  这个说明文件。

## 这个轻量包为什么存在

完整 `LongBench` 体量较大，而这里保留的是一个便于快速 smoke test 的轻量切片，适合：

- 检查模型长上下文处理能力
- 快速跑守护型泛化测试
- 在不展开完整大基准的情况下做日常回归验证
