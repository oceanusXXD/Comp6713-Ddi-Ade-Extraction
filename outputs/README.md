# Outputs

该目录存放训练和分析生成的输出文件。

## 当前结构

- `qwen3_8b_lora/final_adapter/`
  最终 LoRA 适配器目录，可用于推理和发布。
- `qwen3_8b_lora/observability/`
  训练与数据统计相关的观测文件，例如运行环境、配置快照和数据集统计信息。

## 说明

- `final_adapter/` 是需要保留的最终模型产物。
- `observability/` 属于可再生成分析文件。
- checkpoint、临时结果和其他中间输出默认不应长期保留在仓库中。
- `final_adapter/` 下的 `.safetensors` 文件已通过 Git LFS 跟踪。

更详细的模型说明见 `outputs/qwen3_8b_lora/final_adapter/README.md`。
