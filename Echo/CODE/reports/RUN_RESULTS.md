# Run Results

Verification performed for the cleaned `Echo/CODE` package.

## Structural Checks

- The package contains the required top-level folders and main code subfolders.
- The canonical train and inference configs point to files that exist inside `Echo/CODE`.
- The retained adapter exists at `outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter/`.

## Command Checks

The following commands were validated as the expected package entrypoints:

```bash
python -B scripts/train/train_finetune.py --help
python -B scripts/inference/predict.py --help
python -B scripts/evaluation/evaluate_predictions.py --help
python -B -m py_compile scripts/train/train_finetune.py scripts/inference/predict.py scripts/evaluation/evaluate_predictions.py src/data_utils.py src/inference_backends.py src/inference_config.py src/model_utils.py src/observability.py src/parser.py src/prompting.py src/runtime_env.py
```

## Notes

- Full end-to-end training and inference are not executed during package verification because the base model weights are not bundled inside `models/Qwen3-8B`.
- The included adapter, data, and config files are sufficient for another user to rerun the mainline once the base model is supplied locally.
