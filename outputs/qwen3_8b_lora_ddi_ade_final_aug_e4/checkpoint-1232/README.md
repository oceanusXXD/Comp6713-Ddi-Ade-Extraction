# rsLoRA Checkpoint 1232

这个目录保存当前主线训练完成时的最终 checkpoint。

## 定位

- 属于 `outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/` 的最终训练阶段快照
- 2026-03-27 benchmark 中，`ade_corpus_v2` 等 ADE 迁移评测表现优于较早 checkpoint

## 与其它目录的关系

- `checkpoint-620`
  更适合看 DDI 迁移对比
- `checkpoint-1232`
  更接近训练完成版本
- `final_adapter/`
  是最终对外使用的导出 adapter

## 注意

- 如果只是要跑默认推理，优先使用 `../final_adapter/`。
- 如果要复盘训练末期表现，这个目录是最直接的检查点。
