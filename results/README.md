# 结果目录说明


- `README.md`
- `benchmark_suite_latest_raw_clean_base_20260329/`
- `benchmark_suite_latest_raw_clean_balanced_e3_ckpt993_20260329/`

这两个 benchmark 目录分别对应：

- `benchmark_suite_latest_raw_clean_base_20260329/`
  同一套全量 benchmark 上的基座模型结果。
- `benchmark_suite_latest_raw_clean_balanced_e3_ckpt993_20260329/`
  最新清洗数据上训练得到的最佳 rsLoRA 模型结果。

## 如何看结果

- 先看各目录下的 `summary.csv`
  这是整套 benchmark 的总表。
- 再看子目录里的 `*_metrics.json` 和 `*_metrics.txt`
  这是单个数据集的详细指标。
- `*_predictions.jsonl`
  保存逐样本预测，便于抽查错误样本。

## 全量 benchmark 的任务类型

整套 benchmark 在脚本 [run_benchmark_suite.py](/home/coder/data/Comp6713-Ddi-Ade-Extraction/scripts/evaluation/run_benchmark_suite.py) 里注册，任务分为 3 类：

- `labeled_task`
  有金标关系，评估抽取准确率。
- `empty_guardrail`
  文本不属于本任务域，金标固定为空列表 `[]`，用来看模型会不会乱抽。
- `unlabeled_schema`
  没有公开金标关系，只检查输出是否能解析成合法 JSON，以及是否乱输出。

## 每个数据集在做什么

| 数据集 | 任务类型 | 主要来源 | 金标怎么构造 | 结果重点看什么 |
| --- | --- | --- | --- | --- |
| `own_validation` | `labeled_task` | 当前仓库清洗后的验证集 | 直接读取 ChatML 里 assistant 的 JSON 关系 | `exact_match_accuracy`、`precision`、`recall`、`f1` |
| `own_test` | `labeled_task` | 当前仓库清洗后的测试集 | 同上 | `exact_match_accuracy`、`precision`、`recall`、`f1` |
| `seen_style_validation` | `labeled_task` | `evaluate_datasets/seen_style_core/official_held_out` | 同样是 ChatML 金标 | `exact_match_accuracy`、`precision`、`recall`、`f1` |
| `seen_style_test` | `labeled_task` | `evaluate_datasets/seen_style_core/official_held_out` | 同上 | `exact_match_accuracy`、`precision`、`recall`、`f1` |
| `ade_corpus_v2` | `labeled_task` | ADE 迁移集 | 同一段文本里的 `drug` 和 `effect` 聚合成 `ADE` 关系 | `f1` 和 `ADE` 标签表现 |
| `phee_dev` | `labeled_task` | PHEE 开发集 | 只取 `Adverse_event`，把 `Treatment.Drug` 和 `Effect` 两两配对成 `ADE` | `f1` 和 `ADE` 标签表现 |
| `phee_test` | `labeled_task` | PHEE 测试集 | 同上 | `f1` 和 `ADE` 标签表现 |
| `ddi2013_test` | `labeled_task` | DDIExtraction 2013 | XML 中 `ddi=true` 的药物对映射到 `DDI-MECHANISM / EFFECT / ADVISE / INT` | `f1` 和各 DDI 标签表现 |
| `tac2017_adr_gold` | `labeled_task` | TAC 2017 ADR gold XML | 每个文档以 `drug` 为头实体，`AdverseReaction` mention 为尾实体，按 section 切块 | `f1`、`exact_match_accuracy` |
| `cadec_meddra` | `labeled_task` | CADEC forum 数据 | 药名来自文件名，`.ann` 中的反应 mention 组成 `ADE` | `f1`、`exact_match_accuracy` |
| `ifeval_input` | `empty_guardrail` | IFEval | 金标固定为空列表 `[]` | `exact_match_accuracy`、`predicted_nonempty_rate` |
| `docred_dev` | `empty_guardrail` | DocRED dev | 金标固定为空列表 `[]` | `exact_match_accuracy`、`predicted_nonempty_rate` |
| `docred_test` | `empty_guardrail` | DocRED test | 金标固定为空列表 `[]` | `exact_match_accuracy`、`predicted_nonempty_rate` |
| `longbench_multifieldqa_en` | `empty_guardrail` | LongBench | 金标固定为空列表 `[]` | `exact_match_accuracy`、`predicted_nonempty_rate` |
| `longbench_multifieldqa_zh` | `empty_guardrail` | LongBench | 金标固定为空列表 `[]` | `exact_match_accuracy`、`predicted_nonempty_rate` |
| `longbench_passage_retrieval_en` | `empty_guardrail` | LongBench | 金标固定为空列表 `[]` | `exact_match_accuracy`、`predicted_nonempty_rate` |
| `longbench_passage_retrieval_zh` | `empty_guardrail` | LongBench | 金标固定为空列表 `[]` | `exact_match_accuracy`、`predicted_nonempty_rate` |
| `longbench_gov_report` | `empty_guardrail` | LongBench | 金标固定为空列表 `[]` | `exact_match_accuracy`、`predicted_nonempty_rate` |
| `longbench_vcsum` | `empty_guardrail` | LongBench | 金标固定为空列表 `[]` | `exact_match_accuracy`、`predicted_nonempty_rate` |
| `tac2018_test1` | `unlabeled_schema` | TAC 2018 DDI | 只抽 section 文本，不构造金标关系 | `parse_success_rate`、`predicted_nonempty_rate` |
| `tac2018_test2` | `unlabeled_schema` | TAC 2018 DDI | 同上 | `parse_success_rate`、`predicted_nonempty_rate` |

## 指标计算

评估逻辑在 [parser.py](/home/coder/data/Comp6713-Ddi-Ade-Extraction/src/parser.py) 的 `evaluate_prediction_rows` 和 [run_benchmark_suite.py](/home/coder/data/Comp6713-Ddi-Ade-Extraction/scripts/evaluation/run_benchmark_suite.py) 的 `summarize_rows`。

### 1. 解析成功率

- `parse_success_rate = parsed_samples / total_samples`
- 只要模型输出最终能被解析成合法 JSON 关系列表，就算解析成功。

### 2. 完全匹配准确率

- `exact_match_accuracy = exact_match_count / total_samples`
- 对每条样本，把预测关系集合和金标关系集合做集合比较。
- 只有整条样本的关系集合完全一样才算命中。

### 3. Micro Precision / Recall / F1

关系会先被规范化成三元组：

- `(relation_type, head_entity.casefold(), tail_entity.casefold())`

然后在整份数据集上累计：

- `TP = 预测集合 ∩ 金标集合`
- `FP = 预测集合 - 金标集合`
- `FN = 金标集合 - 预测集合`
- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1 = 2PR / (P + R)`

这里要求三件事同时对才算命中：

- 关系标签对
- 头实体对
- 尾实体对

### 4. Per-label 指标

对 5 个规范标签分别统计一遍：

- `ADE`
- `DDI-MECHANISM`
- `DDI-EFFECT`
- `DDI-ADVISE`
- `DDI-INT`

每个标签都有自己的 `tp / fp / fn / precision / recall / f1`。

### 5. Empty Guardrail 任务怎么看

`empty_guardrail` 任务虽然也走 `evaluate_prediction_rows`，但金标永远是空列表 `[]`，所以真正有意义的是：

- `exact_match_accuracy`
  在这类任务里，它就等于“空输出准确率”。
- `predicted_nonempty_rate`
  有多少比例的样本被模型错误地抽出了关系。
- `mean_predicted_relations`
  每条样本平均乱抽了多少条关系。

这类任务里的 `micro f1` 不适合作为主指标，因为金标本来就是空。

### 6. Unlabeled Schema 任务怎么看

`tac2018_test1` 和 `tac2018_test2` 没有公开金标关系，所以不会算 `EM / P / R / F1`，只会统计：

- `parse_success_rate`
- `predicted_nonempty_rate`
- `mean_predicted_relations`

它们的作用是看：

- 输出格式稳不稳
- 模型会不会在未知域文本上大规模乱抽

## 备注

如果是比较最佳 rsLoRA 和 base：

1. 先看两个目录下的 `summary.csv`
2. 再看 `own_validation`、`own_test`、`ddi2013_test`、`ade_corpus_v2`
3. 最后看 `ifeval_input`、`docred_*`、`longbench_*` 判断泛化时是否乱抽
