# Configs

This folder contains the canonical configuration files for the submission package.

## Files

- `qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml`: main training config
- `infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml`: matching inference config for the retained adapter
- `infer_gradio_base.yaml`: base-only Gradio demo config
- `infer_gradio_balanced_e3.yaml`: Gradio demo config with the retained adapter

## Main Paths

Training config:

- base model: `models/Qwen3-8B`
- train split: `../MISC/data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/merged_chatml_train.jsonl`
- validation split: `../MISC/data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/merged_chatml_validation.jsonl`
- output dir: `../MISC/outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3`

Inference config:

- base model: `models/Qwen3-8B`
- adapter: `../MISC/outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter`
- input split: `../MISC/data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/merged_chatml_validation.jsonl`
- output predictions: `../MISC/results/inference_runs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3_validation_predictions.jsonl`

Gradio demo configs:

- base demo: `configs/infer_gradio_base.yaml`
- LoRA demo: `configs/infer_gradio_balanced_e3.yaml`

## Usage

- `scripts/train/train_finetune.py` reads the training config.
- `scripts/inference/predict.py` reads the inference config.
- `scripts/gradio/app.py` reads the Gradio demo configs listed above.

Keep config paths and README references aligned if you move files later.
