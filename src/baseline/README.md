# 规则基线目录说明

这个目录实现了一个不依赖大模型训练的规则式基线，用来给主线 LoRA / rsLoRA 模型提供可解释、可复现的下界参考。

## 文件一览

- `build_lexicons.py`
  从训练数据构建药物词典和不良反应词典。
- `rule_config.py`
  定义基线规则的参数项与默认配置。
- `tune_baseline.py`
  在验证集上搜索较优规则参数。
- `run_baseline.py`
  使用词典和规则在指定数据集上生成预测结果。
- `README.md`
  当前目录说明。

## 依赖的资源与输出目录

- `resources/baseline/`
  保存 `drug_lexicon.json` 和 `effect_lexicon.json`
- `outputs/baseline/`
  保存最佳配置、预测文件和指标文件

## 典型流程

### 1. 构建词典

```bash
.venv/bin/python -m src.baseline.build_lexicons \
  --input_path data/merged_chatml_train.jsonl \
  --output_dir resources/baseline
```

### 2. 在验证集上调参

```bash
.venv/bin/python -m src.baseline.tune_baseline \
  --input_path data/merged_chatml_validation.jsonl \
  --drug_lexicon_path resources/baseline/drug_lexicon.json \
  --effect_lexicon_path resources/baseline/effect_lexicon.json \
  --output_path outputs/baseline/best_config.json
```

### 3. 生成验证集或测试集预测

```bash
.venv/bin/python -m src.baseline.run_baseline \
  --input_path data/merged_chatml_test.jsonl \
  --drug_lexicon_path resources/baseline/drug_lexicon.json \
  --effect_lexicon_path resources/baseline/effect_lexicon.json \
  --config_path outputs/baseline/best_config.json \
  --output_path outputs/baseline/baseline_test_preds.jsonl
```

## 适用场景

- 想快速做 sanity check
- 想确认任务下界是否合理
- 想排查问题究竟出在模型、数据还是评估链路

它不是当前主线方案，但很适合做最小可复现实验。
