# rsLoRA Checkpoint 620

这个目录保存当前主线训练过程中的 `checkpoint-620`。

## 定位

- 属于 `outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/` 训练过程中的阶段性保留点
- 在 2026-03-27 benchmark 中，对 `ddi2013_test` 的迁移表现是当前保留 rsLoRA checkpoint 里最强的一档

## 适合什么时候用

- 想回看中段 checkpoint 的 DDI 迁移表现
- 想和最终 `checkpoint-1232` 做对比
- 想复查训练过程中“泛化更强但未必是最终导出版本”的状态

## 注意

- 这不是默认推理时使用的 adapter 目录。
- 默认推理仍优先加载 `../final_adapter/`。
