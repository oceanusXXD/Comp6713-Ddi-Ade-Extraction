---
base_model: Qwen/Qwen3-8B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen3-8B
- lora
- medical-relation-extraction
- transformers
---

# 旧版可分享 LoRA Adapter 说明

这个目录保存了一套较早期但可直接加载的 LoRA adapter 产物。它不是当前主线训练目录，但仍然是一个完整、可用、可分享的 adapter 示例。

## 当前文件说明

- `adapter_model.safetensors`
  LoRA 权重本体。
- `adapter_config.json`
  LoRA 配置文件。
- `tokenizer.json`
  tokenizer 主配置。
- `tokenizer_config.json`
  tokenizer 补充配置。
- `special_tokens_map.json`
  特殊 token 映射。
- `added_tokens.json`
  额外添加 token 的定义。
- `merges.txt`
  BPE merge 规则。
- `vocab.json`
  词表文件。
- `chat_template.jinja`
  推理时使用的 chat template。
- `README.md`
  这个目录说明。

## 这个目录和当前主线的关系

- 这是旧版稳定 adapter，适合当作“已保留产物”来看
- 当前主线训练配置已经切换到：
  `configs/qwen3_8b_lora_ddi_ade_final.yaml`
- 当前主线输出目录是：
  `outputs/qwen3_8b_lora_ddi_ade_final_aug_e4/`

## 怎么使用它

可以直接在仓库里指定这个 adapter 路径运行推理：

```bash
.venv/bin/python scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_final.yaml \
  --adapter-path outputs/qwen3_8b_lora/final_adapter
```

如果需要，也可以再覆盖 `--base-model` 或配置文件里的基座模型路径。

## 适合什么时候看这个目录

- 你想知道一个完整可分享 adapter 目录里应该包含哪些文件
- 你想快速加载一个现成 adapter 做推理验证
- 你想对照当前主线输出目录检查最终导出产物是否齐全
