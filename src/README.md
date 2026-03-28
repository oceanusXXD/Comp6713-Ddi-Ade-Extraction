# 核心源码目录说明

这个目录放训练、推理、解析、配置加载和辅助工具的核心 Python 模块。

## 顶层模块

- `data_utils.py`
  数据读取、样本解析、ChatML 构造、训练数据编码和部分统计逻辑。
- `model_utils.py`
  训练配置加载、模型与 tokenizer 构建、LoRA 相关初始化和训练工具函数。
- `inference_config.py`
  推理配置读取、命令行覆盖、路径解析和配置校验。
- `inference_backends.py`
  推理后端实现，统一封装 `transformers` 与 `vllm`。
- `prompting.py`
  prompt 相关工具，包括系统提示词加载和 chat template 处理。
- `parser.py`
  模型输出解析、标签规范化、JSON 校验和评估逻辑。
- `observability.py`
  运行时记录工具，用于写环境信息、配置快照、统计文件和训练指标日志。

## 子目录

### `baseline/`

规则基线系统，不依赖 LoRA 训练。

- `build_lexicons.py`
- `rule_config.py`
- `run_baseline.py`
- `tune_baseline.py`
- `README.md`

### `prevalidation/`

历史预验证 / 零样本快速试跑工具，不是当前主线流水线。

- `run_pretest_hf.py`
- `run_pretest_vllm.py`
- `summarize_pretest.py`
- `preview_chatml.py`
- `prompt.txt`
- `README.md`

## 与脚本入口的关系

- `scripts/train/train_finetune.py`
  主要依赖 `model_utils.py`、`data_utils.py`、`observability.py`
- `scripts/inference/predict.py`
  主要依赖 `inference_config.py`、`inference_backends.py`、`parser.py`、`prompting.py`
- `scripts/evaluation/evaluate_predictions.py`
  主要依赖 `parser.py`
- `scripts/analysis/*`
  主要依赖 `data_utils.py`、`parser.py` 和标准库工具

## 维护建议

- 改训练逻辑：优先看 `model_utils.py` 和 `data_utils.py`
- 改推理逻辑：优先看 `inference_backends.py` 和 `inference_config.py`
- 改输出解析或指标：优先看 `parser.py`
- 改运行记录：优先看 `observability.py`

如果只是维护当前主线，不建议优先从 `baseline/` 或 `prevalidation/` 入手。
