# Prompt 目录说明

这个目录保存训练和推理共用的提示词文件。

## 当前文件

- `medical_relation_extraction_system_prompt.txt`
  当前主线系统提示词，用来约束模型输出医疗关系抽取所需的 JSON 格式和标签集合。

## 谁会使用它

- `configs/qwen3_8b_lora_ddi_ade_final.yaml`
- `configs/infer_qwen3_8b_lora_ddi_ade_final.yaml`
- `scripts/inference/predict.py`

## 什么时候改这里

- 要调整任务说明
- 要补充输出格式约束
- 要修改标签解释

如果改了 prompt，建议同时记录影响的训练配置和推理配置，避免训练时和推理时读到不同版本的任务说明。
