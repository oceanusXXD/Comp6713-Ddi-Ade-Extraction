# Prompt 目录说明

这个目录保存训练和推理会共用的提示词文件。

## 文件说明

- `medical_relation_extraction_system_prompt.txt`
  当前主线系统提示词。训练和推理都会读取这份 prompt，用来约束模型输出医疗关系抽取任务所需的 JSON 格式和标签集合。

## 谁会用到它

- `configs/qwen3_8b_lora_ddi_ade_final.yaml`
  训练配置会通过 `system_prompt_path` 引用它。
- `configs/infer_qwen3_8b_lora_ddi_ade_final.yaml`
  推理配置也会通过 `system_prompt_path` 引用它。
- `scripts/inference/predict.py`
  运行时会把它拼到推理输入中。

如果你要改任务说明、输出约束或标签解释，通常从这里改起。
