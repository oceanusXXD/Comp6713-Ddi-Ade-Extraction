# 资源目录说明

这个目录保存不会频繁改动、但会被脚本直接读取的辅助资源。

## 当前文件说明

### `baseline/`

- `baseline/drug_lexicon.json`
  规则基线抽取药物 mention 时使用的词典。
- `baseline/effect_lexicon.json`
  规则基线抽取不良反应 mention 时使用的词典。

## 这些文件由谁生成

- `src.baseline.build_lexicons`
  会根据训练集生成上面这两个词典文件。

## 谁会消费这些文件

- `src.baseline.tune_baseline`
- `src.baseline.run_baseline`

如果你删掉这些文件，规则基线就需要重新先跑一遍词典构建步骤。
