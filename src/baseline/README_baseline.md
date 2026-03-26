# Baseline README

## Overview

This baseline implements a simple rule-based extraction system for the medical relation extraction task. It is designed as a traditional lower-bound method and follows the idea of:

- **spaCy** for sentence splitting and lightweight text processing
- **lexicon matching** for drugs and ADE effects
- **trigger-based rules** for ADE and DDI prediction
- **automatic tuning on validation set** before final testing

The baseline keeps the pipeline simple and interpretable, while producing the same JSONL output format as the LLM scripts so that the same evaluator can be reused.

## Baseline idea

### Core method

The baseline predicts relations in three stages:

1. **Build lexicons from the training set**
   - extract drug mentions
   - extract ADE effect mentions

2. **Tune rules on the validation set**
   - try multiple rule configurations automatically
   - choose the config with the best validation F1

3. **Run once on the test set**
   - fix the selected config
   - generate final baseline predictions
   - evaluate final metrics

### Rule logic

- **ADE**
  - detect drug mentions and effect mentions in the same sentence
  - require ADE trigger words to appear in the sentence
  - each effect is matched only with the nearest drug
  - token distance constraints are used to reduce false matches

- **DDI**
  - detect multiple drug mentions in the same sentence
  - require DDI trigger words to appear
  - only adjacent drug pairs are considered
  - subtype is assigned using subtype-specific trigger groups

This design keeps the baseline lightweight while avoiding the worst over-generation problems of naive all-pair matching.

## Scripts

### `src/baseline/build_lexicons.py`

Uses the training set to build:

- `drug_lexicon.json`
- `effect_lexicon.json`

This corresponds to the train stage of the baseline. It does not train model parameters; it builds lexical resources.

### `src/baseline/tune_baseline.py`

Uses the validation set to automatically search over several rule settings, such as:

- minimum mention length
- maximum number of drugs per sentence for DDI
- additional trigger combinations
- token distance constraints

It outputs:

- `outputs/baseline/best_config.json`

This corresponds to the validation stage.

### `src/baseline/run_baseline.py`

Runs the baseline on a given split (`validation` or `test`) using:

- a drug lexicon
- an effect lexicon
- a config file

It outputs predictions in the same format as the LLM pipeline:

```json
{
  "sample_id": "sample_0001",
  "text": "...",
  "gold_relations": [...],
  "raw_output": "[...]",
  "parsed_output": [...],
  "json_valid": true
}
```

## Full workflow

### Step 1. Train stage: build lexicons from train set

```bash
python -m src.baseline.build_lexicons \
  --input_path data/merged_chatml_train.jsonl \
  --output_dir resources/baseline
```

After that, check the generated files:

```bash
ls -lh resources/baseline/
```

You should see:

- `drug_lexicon.json`
- `effect_lexicon.json`

### Step 2. Validation stage: automatically tune baseline rules

```bash
python -m src.baseline.tune_baseline \
  --input_path data/merged_chatml_validation.jsonl \
  --drug_lexicon_path resources/baseline/drug_lexicon.json \
  --effect_lexicon_path resources/baseline/effect_lexicon.json \
  --output_path outputs/baseline/best_config.json
```

After tuning, you can inspect the selected configuration:

```bash
cat outputs/baseline/best_config.json
```

### Step 3. Validation stage: run baseline again using best config

```bash
python -m src.baseline.run_baseline \
  --input_path data/merged_chatml_validation.jsonl \
  --drug_lexicon_path resources/baseline/drug_lexicon.json \
  --effect_lexicon_path resources/baseline/effect_lexicon.json \
  --config_path outputs/baseline/best_config.json \
  --output_path outputs/baseline/baseline_validation_preds.jsonl
```

Evaluate the validation predictions:

```bash
python -m src.prevalidation.summarize_pretest \
  --pred_path outputs/baseline/baseline_validation_preds.jsonl
```

### Step 4. Test stage: run final baseline on the test set

```bash
python -m src.baseline.run_baseline \
  --input_path data/merged_chatml_test.jsonl \
  --drug_lexicon_path resources/baseline/drug_lexicon.json \
  --effect_lexicon_path resources/baseline/effect_lexicon.json \
  --config_path outputs/baseline/best_config.json \
  --output_path outputs/baseline/baseline_test_preds.jsonl
```

---

### Step 5. Test stage: final evaluation

```bash
python -m src.prevalidation.summarize_pretest \
  --pred_path outputs/baseline/baseline_test_preds.jsonl
```

## Which files matter most

### Main prediction file

- `outputs/baseline/baseline_test_preds.jsonl`

This is the final baseline output on the test set.

### Main config file

- `outputs/baseline/best_config.json`

This is the tuned rule configuration selected on the validation set.

### Main evaluation command

+ `outputs/baseline/baseline_test_preds_metrics.txt`

Final metrics are printed by:

```bash
python -m src.prevalidation.evaluate_predictions \
  --pred_path outputs/baseline/baseline_test_preds.jsonl
```

## Main metrics to report

The most important metrics are:

- **Exact match accuracy**
  - whether all relations in a sample are predicted exactly correctly
- **Precision**
  - among predicted relations, how many are correct
- **Recall**
  - among gold relations, how many are recovered
- **F1**
  - the main overall comparison metric for the baseline versus LLM methods

## Iteration and improvement process

The baseline was not finalized in one step. The first version used a broader matching strategy built from training-set lexicons and simple trigger rules. In that version, the system tended to over-generate relations, especially when a sentence contained multiple drugs or multiple possible effects. This led to too many false positives and low precision.

To improve the baseline, several changes were introduced during the validation stage:

- tightened the trigger lists by removing overly broad patterns
- added extra rule parameters into the config so they could be tuned automatically
- reduced over-pairing by avoiding all-pair matching
- used nearest-drug matching for ADE
- used adjacent-drug pairing for DDI
- added token-distance constraints to filter unlikely matches
- updated the tuning script so that different parameter combinations could be tested automatically on the validation set

These changes made the baseline more conservative and substantially reduced noisy predictions. In practice, this improved the overall validation performance and produced a more reasonable final lower-bound system before running on the test set.

## Expected behavior

This baseline is not expected to perform best. Its purpose is to provide a simple and interpretable lower bound.

Typical expectations:

- JSON validity should be very high
- precision and recall will usually be limited
- the system may still miss implicit relations or make errors in long multi-entity sentences

This is acceptable and useful, because the baseline is meant to highlight the gap between traditional rule-based methods and stronger LLM-based approaches.
