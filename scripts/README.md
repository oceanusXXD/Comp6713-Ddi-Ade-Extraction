# Scripts

该目录存放命令行入口脚本，按功能分组。

## 目录

- `train/train_finetune.py`
  用于 LoRA 微调与 dry run 检查。
- `inference/predict.py`
  用于批量推理与单条文本推理。
- `evaluation/evaluate_predictions.py`
  用于评估预测结果，输出文本和 JSON 格式指标。
- `analysis/analyze_dataset.py`
  用于统计数据集长度、token 分布等信息。

## 使用说明

所有脚本均从仓库根目录执行，例如：

```bash
.venv/bin/python scripts/inference/predict.py --config configs/infer_qwen3_8b.yaml
```

更多命令示例见根目录 `README.md`。
