# 当前主线最终 Adapter

这个目录保存当前主线训练完成后导出的最终 adapter，可直接作为推理时的 `adapter_path` 使用。

## 来源

- 训练配置：`configs/qwen3_8b_lora_ddi_ade_final.yaml`
- 基座模型：`../models/Qwen3-8B`
- 训练输出目录：`outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/`

## 常见文件

- `adapter_model.safetensors`
  LoRA 权重本体。
- `adapter_config.json`
  Adapter 配置。
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `chat_template.jinja`
- `added_tokens.json`
- `merges.txt`
- `vocab.json`

## 如何使用

```bash
.venv/bin/python scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_final.yaml \
  --adapter-path outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/final_adapter
```

## 说明

- 这是“当前主线默认应加载的 adapter”。
- 如果要对比历史版本，请回到 `outputs/README.md` 查其它目录定位。
