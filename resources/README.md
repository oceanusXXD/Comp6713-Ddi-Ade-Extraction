# 资源目录说明

这个目录保存不会频繁改动、但会被脚本直接读取的辅助资源。

## 当前内容

### `baseline/`

- `baseline/drug_lexicon.json`
  规则基线抽取药物 mention 时使用的词典。
- `baseline/effect_lexicon.json`
  规则基线抽取不良反应 mention 时使用的词典。

## 这些文件由谁生成

- `src.baseline.build_lexicons`

## 谁会消费这些文件

- `src.baseline.tune_baseline`
- `src.baseline.run_baseline`

## 注意事项

- 删除这些文件后，规则基线需要重新先跑一遍词典构建。
- 这类资源应尽量保持“脚本生成、README 说明、输出目录消费”三者一致。
