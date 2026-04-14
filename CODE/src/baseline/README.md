# Rule-Based Baseline

Lexicon and rule baseline for ADE/DDI extraction.

## Files

- `build_lexicons.py`: build drug and effect lexicons
- `rule_config.py`: baseline rule settings
- `tune_baseline.py`: search validation settings
- `run_baseline.py`: generate predictions

## Inputs and outputs

- data: `data/merged_chatml_train.jsonl`, `data/merged_chatml_validation.jsonl`, `data/merged_chatml_test.jsonl`
- resources: `resources/baseline/drug_lexicon.json`, `resources/baseline/effect_lexicon.json`
- outputs: `outputs/baseline/`

Install the spaCy model before running the baseline:

```bash
python -m spacy download en_core_web_sm
```

## Commands

Build lexicons:

```bash
python -m src.baseline.build_lexicons \
  --input_path data/merged_chatml_train.jsonl \
  --output_dir resources/baseline
```

Tune on the validation split:

```bash
python -m src.baseline.tune_baseline \
  --input_path data/merged_chatml_validation.jsonl \
  --drug_lexicon_path resources/baseline/drug_lexicon.json \
  --effect_lexicon_path resources/baseline/effect_lexicon.json \
  --output_path outputs/baseline/best_config.json
```

Generate predictions:

```bash
python -m src.baseline.run_baseline \
  --input_path data/merged_chatml_test.jsonl \
  --drug_lexicon_path resources/baseline/drug_lexicon.json \
  --effect_lexicon_path resources/baseline/effect_lexicon.json \
  --config_path outputs/baseline/best_config.json \
  --output_path outputs/baseline/baseline_test_preds.jsonl
```
