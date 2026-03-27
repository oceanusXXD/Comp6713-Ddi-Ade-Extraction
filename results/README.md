# 结果目录说明

这个目录保存推理预测文件、指标文件和变体实验结果。它更像“实验记录区”，不是当前主线唯一输出源。

## 今日主线测试入口

如果你现在要回看 2026-03-27 这轮全量评测，优先看：

- `benchmark_suite_vllm_batch64_20260327/`

这个目录保存了今天串行跑完的一整套 vLLM benchmark，覆盖：

- `base`
- `base+lora`
- `base+rslora@620`
- `base+rslora@930`
- `base+rslora@1232`

建议按下面顺序查看：

- `benchmark_suite_vllm_batch64_20260327/summary.csv`
  全量汇总表，适合做横向比较和后续画表。
- `benchmark_suite_vllm_batch64_20260327/benchmark_analysis_zh.txt`
  中文分析摘要，直接给出主任务、泛化和 guardrail 结论。
- `benchmark_suite_vllm_batch64_20260327/<variant>/`
  每个模型变体对应一个子目录，里面保留逐数据集的 `predictions.jsonl`、`metrics.json` 和 `metrics.txt`。

这轮 benchmark 当前应视为 `results/` 下最重要的一组结果。

## 当前保留原则

- `results/` 顶层默认只长期保留 `README.md` 和 `*.txt`
- 顶层的 `json`、`jsonl`、非 README 的 `md` 结果文件已经清理
- `variant_benchmark_*` 子目录主要保留实验结构、日志、运行配置和文本指标
- benchmark 子目录里的 `*_validation_predictions.jsonl` 与 `*_validation_metrics.json` 已清理，优先保留 `*.txt`

## 顶层文件说明

- `NousResearch_Meta-Llama-3-8B-Instruct_preds_metrics.txt`
  早期 `Meta-Llama-3-8B-Instruct` 预验证或基线对比指标。
- `Qwen_Qwen3-4B-Instruct-2507_preds_metrics.txt`
  `Qwen3-4B-Instruct-2507` 的指标记录。
- `Qwen_Qwen3-8B_preds_metrics.txt`
  `Qwen3-8B` 的指标记录。

## 与增强样本相关的结果

- `qwen3_8b_lora_ddi_ade_c5fc8c06_augmentations_metrics.txt`
  增强样本集合上的文本指标摘要。

## 与 vLLM batch 设置对比相关的结果

当前顶层主要保留：

- `test_metrics_vllm_batch64_*.txt`
  不同设置下的测试集文本指标。

这些文件更适合看“单一测试集上的设置差异”，不再是今天全量 benchmark 的主入口。

常见后缀含义：

- `base`：只用基座模型
- `lora`：加载 LoRA adapter
- `thinking` / `no_thinking`：是否开启思考模式
- `512` / `1024` / `default2048`：生成长度设置

## `variant_benchmark_*` 子目录说明

这些目录来自 `scripts/experiments/run_qwen3_lora_variant_benchmark.py`，用于比较不同 LoRA 变体。

当前保留的实验目录包括：

- `variant_benchmark_qwen3_8b_full/`
- `variant_benchmark_qwen3_8b_full_final/`
- `variant_benchmark_qwen3_8b_screen1e/`
- `variant_benchmark_qwen3_8b_screen1e_venv/`
- `variant_benchmark_qwen3_8b_screen1e_venv_rest/`

已经清理的冒烟实验目录包括：

- `variant_benchmark_qwen3_8b_envfix_smoke/`
- `variant_benchmark_qwen3_8b_patch_smoke/`
- `variant_benchmark_qwen3_8b_patch_smoke2/`
- `variant_benchmark_qwen3_8b_smoke/`

这类目录里常见文件和子目录：

- `summary.json`
  机器可读的实验汇总，主要作为本地分析中间产物。
- `summary.md`
  人类可读的实验汇总。
- `*_validation_metrics.txt`
  各变体的文本指标，属于优先保留结果。
- `logs/`
  训练和推理日志。
- `runtime_configs/`
  实验当时落盘的运行时配置。
- `outputs/`
  实验运行生成的 adapter、checkpoint 和中间输出。

## 什么时候看这里

- 想回看历史实验结果
- 想知道某个设置下保留下来的文本指标在哪
- 想做模型 / 配置之间的横向比较

如果你只关心“当前主线应该怎么跑”，优先看 `configs/`、`outputs/` 和根目录 `README.md`。
