# 预验证 (Prevalidation) 脚本使用说明

本目录下的脚本主要用于在本地或者单机环境下，使用开源大语言模型（如 Llama-3 等）来进行零样本（Zero-shot）的 DDI/ADE 关系抽取测试与效果评估。

## 1. 批量推理脚本：`run_pretest_vllm.py`

此脚本使用高吞吐量的 `vLLM` 框架加载大模型，并发地执行关系抽取。相较于原生 HuggingFace 管道，能在极短时间（分钟级）内跑完全部测试集。

### 常用命令
```bash
python src/prevalidation/run_pretest_vllm.py \
    --model_name <Hugginface 模型名称或本地路径> \
    --output_path <预测结果输出路径> \
    --limit 0
```

### 可选参数说明
- `--model_name` **(必填)**: Huggingface 上托管的模型名字（例如 `NousResearch/Meta-Llama-3-8B-Instruct`）或者是本地权重文件夹。
- `--output_path` **(必填)**: 生成的 JSONL 文件的保存路径（例如 `results/nous_llama3_preds.jsonl`）。
- `--input_path`: 输入语料路径。默认为 `data/merged_chatml_test.jsonl`。
- `--prompt_path`: System Prompt，也就是任务的提示词文本。默认为 `src/prevalidation/prompt.txt`。
- `--limit`: 测试条数。默认为 `50`，可以用来快速测试脚本是否报错。设为 `0` 或负数将跑完验证集中所有的样本满测。
- `--temperature`: 生成温度参数，对于信息抽取任务，默认且推荐为 `0.0` (贪心/确定性生成)。
- `--max_new_tokens`: 控制模型回答的最大长度，默认为 `256`。

---

## 2. 指标评测与总结脚本：`summarize_pretest.py`

在执行完大模型的批量推理后，通过该脚本计算各类表现指标：包含精确匹配（Exact Match Acc），抽取的精确度（Precision）、召回率（Recall）以及 F1 分数，还会打印部分抽取错位的样本供进一步 Prompt 分析。

### 常用命令
```bash
python src/prevalidation/summarize_pretest.py --pred_path <模型预测生成的JSONL文件路径>
```

### 可选参数说明
- `--pred_path` **(必填)**: 推理脚本输出的预测文件路径（与上一指令的 `--output_path` 对应）。

### 指标解读
- **JSON validity rate**: 表示大模型正确遵循了 JSON 数组格式输出的比例。
- **Exact match accuracy**: 完全预测对一个句子中所有关系（没有漏抽，没有多抽）的句子比例。
- **Precision, Recall, F1**: 微平均（Micro-average）的三大信息抽取指标，是学术界衡量模型能力的黄金标准。
- **Examples of mismatched cases**: 列出错误抽取的案例，方便开发人员 debug 以及进行错误分析（Error Analysis）。
