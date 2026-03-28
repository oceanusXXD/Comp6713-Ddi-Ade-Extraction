# rsLoRA Checkpoint 1085

这个目录保存当前主线训练后段的 `checkpoint-1085`。

## 定位

- 属于 `outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/` 的阶段性中间 checkpoint
- 主要价值是回看训练后段状态，而不是作为默认对外引用版本

## 适合什么时候看

- 想分析训练后段变化趋势
- 想和 `checkpoint-620`、`checkpoint-1232` 做阶段对比
- 想检查某次中间导出的权重与配置是否一致

## 注意

- 默认推理不直接引用这里。
- 对外共享或常规评测，优先使用 `final_adapter/` 或明确指定其它 checkpoint。
