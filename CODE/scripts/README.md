# Scripts

Command-line entrypoints for data prep, training, inference, evaluation, demos, and experiments.

## Subdirectories

- `train/`: `train_finetune.py`
- `inference/`: `predict.py`
- `evaluation/`: `evaluate_predictions.py`, `evaluate_predictions_by_augmentation.py`, `run_benchmark_suite.py`, `quantitative_evaluation.py`
- `analysis/`: `analyze_dataset.py`, `audit_and_prepare_final_dataset.py`, `fetch_evaluate_datasets.py`
- `demo/`: `gradio_demo.py`
- `experiments/`: `run_qwen3_lora_variant_benchmark.py`

## Main flow

`analysis/audit_and_prepare_final_dataset.py` -> `train/train_finetune.py` -> `inference/predict.py` -> `evaluation/evaluate_predictions.py`

## Common commands

Rebuild the processed dataset:

```bash
python scripts/analysis/audit_and_prepare_final_dataset.py
```

Train:

```bash
python scripts/train/train_finetune.py \
  --config configs/qwen3_8b_lora_ddi_ade_final.yaml \
  --do-train
```

Run inference:

```bash
python scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_final.yaml
```

Evaluate predictions:

```bash
python scripts/evaluation/evaluate_predictions.py \
  --predictions-path results/inference_runs/your_predictions.jsonl
```

Evaluate by augmentation source:

```bash
python scripts/evaluation/evaluate_predictions_by_augmentation.py \
  --predictions-path results/inference_runs/your_predictions.jsonl \
  --source-path data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train_augmentations.jsonl
```

Generate the benchmark comparison summary:

```bash
python scripts/evaluation/quantitative_evaluation.py
```
