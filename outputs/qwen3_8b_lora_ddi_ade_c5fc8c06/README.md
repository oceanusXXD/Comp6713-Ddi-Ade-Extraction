# 历史训练输出目录：`c5fc8c06`

这个目录保存一套基于旧数据版本 `c5fc8c06` 训练得到的历史产物。

## 目录内容

- `checkpoint-18/`
  训练中间 checkpoint。
- `final_adapter/`
  这轮历史训练导出的最终 adapter。
- `observability/`
  当时落盘的环境、配置和训练记录。

## 与当前主线的关系

- 它不是当前推荐训练输出目录。
- `results/benchmark_suite_vllm_batch64_20260327/` 中的 `lora` 变体使用的是这里的 `final_adapter/`。

## 适合什么时候看

- 想对比旧数据版本和当前主线之间的差异
- 想回看 benchmark 里的 `base+lora` 实际加载了哪套 adapter
- 想保留一个历史可复现实验基线
