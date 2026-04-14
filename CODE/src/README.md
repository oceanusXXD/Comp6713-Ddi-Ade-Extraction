# Source Modules

Core Python modules for training, inference, parsing, configuration, and runtime logging.

## Files

- `data_utils.py`: dataset loading and ChatML preparation
- `model_utils.py`: training config, model setup, and LoRA helpers
- `inference_config.py`: inference config loading and validation
- `inference_backends.py`: `transformers` and `vllm` backends
- `prompting.py`: prompt loading and chat template helpers
- `parser.py`: output parsing and evaluation logic
- `observability.py`: runtime logging and summaries
- `runtime_env.py`: runtime environment helpers

## Subdirectories

- `baseline/`: lexicon and rule baseline
- `prevalidation/`: older zero-shot and quick-check utilities

## Used by

- `scripts/train/train_finetune.py`: `model_utils.py`, `data_utils.py`, `observability.py`
- `scripts/inference/predict.py`: `inference_config.py`, `inference_backends.py`, `parser.py`, `prompting.py`
- `scripts/evaluation/evaluate_predictions.py`: `parser.py`
- `scripts/analysis/*`: `data_utils.py`, `parser.py`
