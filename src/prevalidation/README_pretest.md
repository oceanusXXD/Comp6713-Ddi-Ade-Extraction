# 预验证目录说明

这个目录保存的是仓库较早期使用的预验证脚本，主要用于零样本 / 少量样本快速试跑，不是当前主线训练流水线。

## 文件说明

- `run_pretest_hf.py`
  使用 Hugging Face 推理后端做预验证。
- `run_pretest_vllm.py`
  使用 vLLM 推理后端做预验证，通常吞吐更高。
- `summarize_pretest.py`
  汇总预验证输出，计算精确匹配、Precision、Recall、F1，并打印部分错误案例。
- `preview_chatml.py`
  预览输入样本，帮助检查 ChatML 格式和 prompt 拼接情况。
- `prompt.txt`
  旧版预验证脚本使用的 prompt 文件。
- `README_pretest.md`
  这个说明文件。

## 这些脚本和当前主线的关系

- 当前主线训练：不用这里的脚本
- 当前主线默认推理：不用这里的脚本
- 历史零样本 / 快速预跑：会用这里的脚本

如果你只是维护当前主线，优先看：

- `scripts/train/train_finetune.py`
- `scripts/inference/predict.py`
- `scripts/evaluation/evaluate_predictions.py`

## 常用命令

### 使用 vLLM 跑预验证

```bash
python src/prevalidation/run_pretest_vllm.py \
  --model_name Qwen/Qwen3-8B \
  --output_path results/pretest_preds.jsonl \
  --limit 50
```

### 汇总预验证结果

```bash
python src/prevalidation/summarize_pretest.py \
  --pred_path results/pretest_preds.jsonl
```

## 适用场景

- 快速试某个开源模型是否能跑通
- 做 prompt 初筛
- 在不进入完整训练流水线的前提下做零样本对比
