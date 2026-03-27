# 规则基线目录说明

这个目录实现了一个不依赖大模型训练的规则式基线，用来给主线 LoRA 模型提供一个可解释、可复现的下界参考。

## 目录里的文件是干什么的

- `build_lexicons.py`
  从训练数据里抽取药物 mention 和不良反应 mention，生成基线词典。
- `rule_config.py`
  定义基线规则用到的参数项和默认配置。
- `tune_baseline.py`
  在验证集上搜索不同规则参数组合，挑出效果较好的配置。
- `run_baseline.py`
  使用词典和规则在指定数据集上生成预测结果。
- `README_baseline.md`
  这个说明文件。

## 资源和输出会落到哪里

运行这个基线时，通常会配合下面两个目录：

- `resources/baseline/`
  保存 `drug_lexicon.json` 和 `effect_lexicon.json`
- `outputs/baseline/`
  保存最佳配置、验证集预测、测试集预测和指标文件

## 典型流程

### 1. 构建词典

```bash
python -m src.baseline.build_lexicons \
  --input_path data/merged_chatml_train.jsonl \
  --output_dir resources/baseline
```

生成文件：

- `resources/baseline/drug_lexicon.json`
- `resources/baseline/effect_lexicon.json`

### 2. 在验证集上调参

```bash
python -m src.baseline.tune_baseline \
  --input_path data/merged_chatml_validation.jsonl \
  --drug_lexicon_path resources/baseline/drug_lexicon.json \
  --effect_lexicon_path resources/baseline/effect_lexicon.json \
  --output_path outputs/baseline/best_config.json
```

### 3. 生成验证集或测试集预测

```bash
python -m src.baseline.run_baseline \
  --input_path data/merged_chatml_test.jsonl \
  --drug_lexicon_path resources/baseline/drug_lexicon.json \
  --effect_lexicon_path resources/baseline/effect_lexicon.json \
  --config_path outputs/baseline/best_config.json \
  --output_path outputs/baseline/baseline_test_preds.jsonl
```

## 这个基线适合什么时候看

- 你想确认任务下界是否合理
- 你想快速验证数据和评估链路是否通
- 你想排查模型到底是“没学会任务”，还是“评估脚本 / 数据有问题”

它不是当前主线方案，但很适合做 sanity check。
