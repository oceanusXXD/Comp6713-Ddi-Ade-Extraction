# 数据集说明

这个目录存放训练、验证、测试所用的 ChatML JSONL 数据。

## 当前文件

- `merged_chatml_train.jsonl`
- `merged_chatml_validation.jsonl`
- `merged_chatml_test.jsonl`

## 数据格式

每一行都是一个 JSON 对象，核心是 `messages` 数组，包含 `system`、`user`、`assistant` 三类消息。

示例：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert medical information extraction system..."
    },
    {
      "role": "user",
      "content": "Erythromycin may increase the serum concentration of simvastatin."
    },
    {
      "role": "assistant",
      "content": "[{\"head_entity\": \"Erythromycin\", \"tail_entity\": \"simvastatin\", \"relation_type\": \"DDI-mechanism\"}]"
    }
  ]
}
```

说明：

- `user` 是待抽取的原始医疗文本。
- `assistant` 必须是一个 JSON 数组字符串。
- 没有关系时，`assistant` 仍然必须输出 `[]`。

## 关系标签

仓库内部统一使用以下 canonical labels：

- `ADE`
- `DDI-MECHANISM`
- `DDI-EFFECT`
- `DDI-ADVISE`
- `DDI-INT`

历史数据里如果出现 `DDI-mechanism`、`DDI-effect` 这类大小写变体，训练和评估脚本会做规范化，但最终输出建议始终使用上面的全大写形式。

## 当前划分

- `merged_chatml_train.jsonl`：训练集
- `merged_chatml_validation.jsonl`：验证集
- `merged_chatml_test.jsonl`：测试集

## 给下一个人的使用方式

如果你只是做推理分析，有两种方式：

1. 把自己的数据文件放在 `data/` 下，然后运行：

```bash
.venv/bin/python scripts/inference/predict.py \
  --config configs/infer_qwen3_8b.yaml \
  --input-path data/your_dataset.jsonl
```

2. 如果你想直接复用默认 split 文件名，就把自己的文件替换成：

- `merged_chatml_validation.jsonl`
- `merged_chatml_test.jsonl`

## 注意事项

- 评估脚本要求预测文件和 gold 数据在样本顺序或 `sample_id` 上能对齐。
- 训练与推理默认使用仓库内统一 prompt，而不是完全依赖原始数据里的 `system` 提示。
- 超过 `max_seq_length` 的样本在训练时会被跳过，而不是截断目标答案。
