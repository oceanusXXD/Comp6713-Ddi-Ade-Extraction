# 配置目录说明

这个目录保存当前主线训练和推理使用的 YAML 配置文件。

## 文件说明

- `qwen3_8b_lora_ddi_ade_final.yaml`
  当前主线训练配置。定义训练集 / 验证集路径、LoRA 参数、batch size、学习率、输出目录等。
- `infer_qwen3_8b_lora_ddi_ade_final.yaml`
  当前主线推理配置。定义基座模型、adapter 路径、推理输入、后端参数和输出路径。

## 这两个文件怎么配合

- 训练阶段：
  `scripts/train/train_finetune.py` 读取 `qwen3_8b_lora_ddi_ade_final.yaml`
- 推理阶段：
  `scripts/inference/predict.py` 读取 `infer_qwen3_8b_lora_ddi_ade_final.yaml`

## 你最常改的字段

### 训练配置里

- `train_path`
  训练集路径。
- `validation_path`
  验证集路径。
- `model_name_or_path`
  基座模型路径。
- `output_dir`
  训练输出目录。
- `lora_r`、`lora_alpha`、`lora_dropout`
  LoRA 关键超参数。
- `learning_rate`、`num_train_epochs`
  训练超参数。

### 推理配置里

- `model.base_model_name_or_path`
  推理时加载的基座模型。
- `model.adapter_path`
  LoRA adapter 路径。
- `data.input_path`
  推理输入文件。
- `backend`
  推理后端，例如 `transformers` 或 `vllm`。
- `output.*`
  预测结果和指标输出路径。
