# 结果目录说明

这个目录保存推理预测、指标文件、benchmark 归档和实验结果。整理后，顶层不再堆放零散的指标文本，摘要指标统一收进了 `metrics/`。

## 现在先看哪里

### 当前最重要的综合 benchmark

- `benchmark_suite_vllm_batch64_20260327/`

这是 2026-03-27 全量 vLLM benchmark 的主归档目录，仍然是结果查看首选入口。

### 汇总型文本指标

- `metrics/model_snapshots/`
  早期模型快照指标。
- `metrics/augmentations/`
  增强样本相关指标。
- `metrics/test_set_sweeps/`
  单测试集设置对比指标。

### 未来默认推理输出

- `inference_runs/`
  默认推理脚本写预测和指标的目录。

### 旧版预验证输出

- `prevalidation/`
  旧版预验证脚本默认写出的预测目录。

## 目录分工

- `benchmark_suite_vllm_batch64_20260327/`
  当前最重要的综合 benchmark 归档。
- `benchmark_serial_vllm_batch64_20260327/`
  串行 benchmark 归档。
- `benchmarks/smoke/`
  smoke benchmark 归档。
- `variant_benchmark_*`
  变体实验结果目录。
- `metrics/`
  顶层摘要型文本指标。
- `inference_runs/`
  未来常规推理输出。
- `prevalidation/`
  未来旧版预验证输出。

## `metrics/` 下的内容

### `metrics/model_snapshots/`

- `NousResearch_Meta-Llama-3-8B-Instruct_preds_metrics.txt`
- `Qwen_Qwen3-4B-Instruct-2507_preds_metrics.txt`
- `Qwen_Qwen3-8B_preds_metrics.txt`

### `metrics/augmentations/`

- `qwen3_8b_lora_ddi_ade_c5fc8c06_augmentations_metrics.txt`

### `metrics/test_set_sweeps/`

这里保存原先散落在顶层的 `test_metrics_vllm_batch64_*.txt` 文件。

常见后缀含义：

- `base`
  只加载基座模型
- `lora`
  加载 LoRA adapter
- `thinking` / `no_thinking`
  是否启用思考模式
- `512` / `1024` / `default2048`
  生成长度设置

## 使用建议

- 想回看完整 benchmark：优先看 `benchmark_suite_vllm_batch64_20260327/`
- 想快速看单模型文本指标：看 `metrics/`
- 想看未来默认推理输出：看 `inference_runs/`
- 想看旧版预验证输出：看 `prevalidation/`
- 想查实验过程：看 `variant_benchmark_*`
