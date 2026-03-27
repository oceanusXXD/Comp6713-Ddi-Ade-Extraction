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
  当前 rsLoRA 主线训练完成后的 `checkpoint-1232` 结果。

## 目录怎么读

- `summary.csv`
  全部数据集、全部模型变体的汇总表，是做论文表格、可视化和横向比较的首选入口。
- `benchmark_analysis_zh.txt`
  中文结论版分析，已经整理出主任务、外部泛化和 guardrail 三方面的判断。
- `<variant>/`
  每个变体单独一个目录，里面按数据集保留：
  - `*_predictions.jsonl`
  - `*_metrics.json`
  - `*_metrics.txt`

## 当前建议

- 如果只追求 ADE 主任务最好结果，优先看 `rslora_1232`。
- 如果要一个更均衡的默认版本，优先看 `rslora_930`。
- 如果重点观察 DDI 泛化，优先看 `rslora_620`。

## 备注

- 这个目录是今天测试内容的主归档入口。
- 顶层 `results/` 下零散的 `test_metrics_vllm_batch64_*.txt` 仍然保留，但它们主要用于单组设置对比，不替代这里的全量 benchmark。
