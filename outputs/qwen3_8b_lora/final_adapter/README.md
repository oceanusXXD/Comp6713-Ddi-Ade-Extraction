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

# Qwen3-8B LoRA Adapter for ADE/DDI Extraction

这个目录是仓库中可直接分享的 LoRA 适配器，基座模型为 `Qwen/Qwen3-8B`。

## 任务

输入一段医疗文本，输出结构化关系：

- `ADE`
- `DDI-MECHANISM`
- `DDI-EFFECT`
- `DDI-ADVISE`
- `DDI-INT`

输出格式为 JSON 数组，每个元素包含：

- `head_entity`
- `tail_entity`
- `relation_type`

## 使用方式

在本仓库中直接通过 `scripts/inference/predict.py` 加载：

```bash
.venv/bin/python scripts/inference/predict.py \
  --config configs/infer_qwen3_8b.yaml \
  --base-model Qwen/Qwen3-8B \
  --adapter-path outputs/qwen3_8b_lora/final_adapter
```

如果你已经把 base model 下载到本地，也可以把 `--base-model` 换成自己的本地目录。

## 说明

- 这个目录应该被保留并可上传。
- `adapter_model.safetensors` 已配置为 Git LFS 跟踪。
- 训练、推理、评估的完整说明见仓库根目录 `README.md`。
