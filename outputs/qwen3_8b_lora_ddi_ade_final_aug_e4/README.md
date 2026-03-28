# 当前主线训练输出目录

这个目录对应训练配置 `configs/qwen3_8b_lora_ddi_ade_final.yaml` 中的 `output_dir`，是当前主线最重要的一组训练产物。

## 目录内容

- `checkpoint-620/`
  训练中段保留 checkpoint；在 2026-03-27 benchmark 中，DDI 迁移表现最强。
- `checkpoint-1085/`
  训练后段中间 checkpoint，用于回看训练过程与阶段性导出。
- `checkpoint-1232/`
  训练完成时的最终 checkpoint。
- `final_adapter/`
  最终导出的可直接加载 adapter。
- `observability/`
  训练配置快照、参数统计、数据统计、环境信息和训练日志。

## 与 benchmark 的对应关系

`results/benchmark_suite_vllm_batch64_20260327/` 中与本目录直接对应的变体有：

- `rslora_620`
- `rslora_1232`

其中：

- `checkpoint-620`
  更适合看 DDI 迁移表现
- `checkpoint-1232`
  是最终训练完成版本，也是默认导出 `final_adapter/` 的来源

`rslora_930` 的结果快照保留在 `results/` 中，但当前没有在这里继续保留对应 checkpoint 目录。

## 使用建议

- 想直接推理：优先用 `final_adapter/`
- 想复查训练中间状态：看 `checkpoint-*`
- 想看训练环境和配置：看 `observability/`
