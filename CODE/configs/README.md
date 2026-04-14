# Configs

YAML configuration files for training, inference, and demo runs.

## Main files

- `qwen3_8b_lora_ddi_ade_final.yaml`: training config
- `infer_qwen3_8b_lora_ddi_ade_final.yaml`: inference config
- `infer_gradio_base.yaml`: base demo config
- `infer_gradio_balanced_e3.yaml`: demo config with adapter

Other `*.yaml` files are experiment variants.

## Scripts that use them

- `scripts/train/train_finetune.py` reads `qwen3_8b_lora_ddi_ade_final.yaml`
- `scripts/inference/predict.py` reads `infer_qwen3_8b_lora_ddi_ade_final.yaml`

## Default paths in the main configs

Training config:

- base model: `models/Qwen3-8B`
- train data: `data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`
- validation data: `data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- output dir: `outputs/qwen3_8b_lora_ddi_ade_final_aug_e4`

Inference config:

- base model: `models/Qwen3-8B`
- adapter: `outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter`
- input: `data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`
- predictions: `results/inference_runs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3_validation_predictions.jsonl`
- metrics: `results/inference_runs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3_validation_metrics.txt`

Demo configs:

- base demo: `configs/infer_gradio_base.yaml`
- adapter demo: `configs/infer_gradio_balanced_e3.yaml`
- adapter path: `outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter`

## Common fields

Training configs:

- `train_path`
- `validation_path`
- `model_name_or_path`
- `output_dir`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- `use_rslora`
- `learning_rate`
- `num_train_epochs`

Inference configs:

- `model.base_model_name_or_path`
- `model.adapter_path`
- `data.input_path`
- `backend`
- `inference.batch_size`
- `inference.max_new_tokens`
- `output.predictions_path`
- `output.metrics_path`
