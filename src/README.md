# 核心源码目录说明

这个目录放的是训练、推理、解析、配置加载和辅助工具的核心 Python 模块。

## 顶层模块说明

- `data_utils.py`
  数据读取、样本解析、ChatML 构造、训练数据编码和部分数据统计逻辑。
- `model_utils.py`
  训练配置加载、模型与 tokenizer 加载、LoRA 相关初始化和训练侧工具函数。
- `inference_config.py`
  推理配置读取、命令行覆盖、路径解析和配置校验。
- `inference_backends.py`
  推理后端实现，统一封装 `transformers` 和 `vllm` 两种执行路径。
- `prompting.py`
  prompt 相关工具，包括系统提示词加载、chat template 处理等。
- `parser.py`
  模型输出解析、关系标签规范化、JSON 校验和评估计算相关逻辑。
- `observability.py`
  运行时记录工具，用于写出环境信息、配置快照、训练过程统计和观测文件。

## `baseline/` 子目录

这个子目录是规则基线系统，不依赖 LoRA 训练。

- `baseline/build_lexicons.py`
  从训练集构建药物词典和不良反应词典。
- `baseline/rule_config.py`
  规则基线使用的配置定义和默认参数。
- `baseline/run_baseline.py`
  用规则基线在验证集或测试集上生成预测结果。
- `baseline/tune_baseline.py`
  在验证集上自动搜索较优的规则参数。
- `baseline/README_baseline.md`
  基线系统详细说明。

## `prevalidation/` 子目录

这个子目录是旧版预验证 / 零样本测试工具，主要用于早期模型快速试跑。

- `prevalidation/run_pretest_hf.py`
  使用 Hugging Face 后端跑预验证。
- `prevalidation/run_pretest_vllm.py`
  使用 vLLM 后端跑预验证。
- `prevalidation/summarize_pretest.py`
  汇总预验证指标并打印错误案例。
- `prevalidation/preview_chatml.py`
  预览 ChatML 样本内容，方便快速检查数据格式。
- `prevalidation/prompt.txt`
  预验证脚本使用的旧版 prompt 文件。
- `prevalidation/README_pretest.md`
  预验证子目录说明。

## 关系图怎么理解

- `scripts/train/train_finetune.py`
  主要依赖 `model_utils.py`、`data_utils.py`、`observability.py`
- `scripts/inference/predict.py`
  主要依赖 `inference_config.py`、`inference_backends.py`、`parser.py`、`prompting.py`
- `scripts/evaluation/evaluate_predictions.py`
  主要依赖 `parser.py`
- `scripts/analysis/*`
  主要依赖 `data_utils.py`、`parser.py` 和若干标准库工具

如果你要改训练逻辑，优先看 `model_utils.py` 和 `data_utils.py`；如果你要改推理或输出解析，优先看 `inference_backends.py`、`inference_config.py` 和 `parser.py`。
