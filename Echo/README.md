# Echo Submission Package

COMP6713 DDI/ADE extraction project.

## Structure

- `README.md`: top-level submission guide
- `CODE/`: reproducible code package with configs, scripts, and Python modules
- `MISC/`: datasets, retained adapter assets, evaluation summaries, and reproducibility notes
- `REPORT.pdf`: final written report
- `PRESENTATION.pdf`: presentation slides
- `CONTRIBUTION.md`: draft contribution statement

## What Is Included

`CODE/` contains the current reproducibility path for:

- LoRA training with `scripts/train/train_finetune.py`
- dataset inference and single-text inference with `scripts/inference/predict.py`
- Gradio demo launch with `scripts/gradio/app.py`
- evaluation with `scripts/evaluation/evaluate_predictions.py`

The retained adapter is stored under:

```text
MISC/outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter/
```

## Before You Run Anything

The package does not bundle the base model weights. The configs expect:

```text
CODE/models/Qwen3-8B
```

Read `CODE/README.md` first for:

- how to check whether the local base model already exists
- how to reproduce the packaged LoRA run
- how to run the base model only
- how to launch the Gradio UI
- how Git LFS is used for `*.safetensors`

## Final Submission Notes

- `REPORT.pdf` and `PRESENTATION.pdf` are included at the top level.
- Update `CONTRIBUTION.md` with the final names, zIDs, and percentages before submission.
