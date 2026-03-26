# Source Modules

该目录存放训练、推理、解析和配置加载所使用的核心模块。

## 主要模块

- `data_utils.py`
  数据读取、样本构建、训练集编码和统计逻辑。
- `model_utils.py`
  训练配置加载、模型与 tokenizer 加载、LoRA 相关工具。
- `inference_config.py`
  推理配置加载、参数覆盖和路径校验。
- `inference_backends.py`
  `transformers` 和 `vllm` 两种推理后端实现。
- `prompting.py`
  prompt 构造和 chat template 处理。
- `parser.py`
  模型输出解析、标签规范化和评估指标计算。
- `observability.py`
  运行环境、训练配置和统计信息写出。

## 预验证脚本

`prevalidation/` 目录包含预验证与旧版测试脚本：

- `run_pretest_hf.py`
- `run_pretest_vllm.py`
- `summarize_pretest.py`
- `preview_chatml.py`

对应说明见 `src/prevalidation/README_pretest.md`。
