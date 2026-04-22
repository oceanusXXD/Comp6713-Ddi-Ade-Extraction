# Configs

This folder contains the canonical configuration files for the submission package.

## Files

- `qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml`: main training config
- `infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml`: matching inference config for the retained adapter

## Main Paths

Training config:

- base model: `models/Qwen3-8B`
- train split: `data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/merged_chatml_train.jsonl`
- validation split: `data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/merged_chatml_validation.jsonl`
- output dir: `outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3`

Inference config:

- base model: `models/Qwen3-8B`
- adapter: `outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter`
- input split: `data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/merged_chatml_validation.jsonl`
- output predictions: `results/inference_runs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3_validation_predictions.jsonl`

## Usage

- `scripts/train/train_finetune.py` reads the training config.
- `scripts/inference/predict.py` reads the inference config.

Keep config paths and README references aligned if you move files later.
