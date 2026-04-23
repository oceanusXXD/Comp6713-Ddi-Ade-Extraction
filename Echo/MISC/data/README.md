# Data

This package includes the processed dataset required to rerun the main training and validation pipeline.

## Included Files

Primary processed dataset:

```text
data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/
```

Included files inside that directory:

- `merged_chatml_train.jsonl`
- `merged_chatml_validation.jsonl`
- `merged_chatml_test.jsonl`
- `manifest.json`
- `clean_stats.json`
- `dedup_stats.json`
- `sanitize_stats.json`

Convenience top-level copies:

- `data/merged_chatml_train.jsonl`
- `data/merged_chatml_validation.jsonl`
- `data/merged_chatml_test.jsonl`

## Dataset Shape

The main pipeline uses ChatML-style JSONL rows with `system`, `user`, and `assistant` messages. The assistant message stores the gold relation list as JSON.

## Mainline Counts

According to the retained observability files:

- train rows: `5287`
- validation rows: `588`
- test rows: `619`

Use the processed `latest_raw_clean` folder as the canonical source for the submission package.
