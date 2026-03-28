# 2026-03-27 全量评测归档

这个目录保存 2026-03-27 使用 vLLM 串行完成的全量 benchmark 结果。

## 评测设置

- 推理后端：`.venv` 中的 vLLM
- 调度方式：串行执行
- batch size：`64`
- `max_new_tokens`：`512`

## 评测对象

- `base`
  纯 `Qwen3-8B` 基座模型。
- `lora`
  `outputs/qwen3_8b_lora_ddi_ade_c5fc8c06/final_adapter`。
- `rslora_620`
  当前 rsLoRA 主线在 `checkpoint-620` 的结果。
- `rslora_930`
  历史 rsLoRA `checkpoint-930` 的结果快照。
- `rslora_1232`
  当前 rsLoRA 主线在 `checkpoint-1232` 的结果。

## 建议先看什么

- `summary.csv`
  全部数据集、全部模型变体的汇总表。
- `benchmark_analysis_zh.txt`
  中文分析摘要。
- `<variant>/`
  每个变体按数据集保留的 `predictions` 与 `metrics`。

## 读表时可先记住的结论

- in-domain 主任务：
  `rslora_930` 在 `rslora_own_test` 上的 F1 为 `0.6784`，是这轮保留结果中最强。
- ADE 迁移：
  `rslora_1232` 在 `ade_corpus_v2` 上的 F1 为 `0.9135`，优于 `rslora_930` 和 `rslora_620`。
- DDI 迁移：
  `rslora_620` 在 `ddi2013_test` 上的 F1 为 `0.4567`，优于 `rslora_930` 和 `rslora_1232`。

## 如何使用这些结论

- 如果优先追求主任务 in-domain 表现：先看 `rslora_930`
- 如果更看重 ADE 迁移：先看 `rslora_1232`
- 如果重点观察 DDI 迁移：先看 `rslora_620`

## 备注

- 这组目录是 2026-03-27 benchmark 的主归档入口。
- 单测试集设置对比文本指标现在统一收在 `results/metrics/test_set_sweeps/`，它们主要用于单组设置对比，不替代这里的全量 benchmark。
