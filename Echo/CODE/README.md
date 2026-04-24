# COMP6713 DDI/ADE Extraction

This package contains the cleaned code submission for ADE and DDI relation extraction from medical text.

## Task

Each model prediction must be a JSON list of relations in the following format:

```json
[
  {
    "head_entity": "string",
    "tail_entity": "string",
    "relation_type": "ADE | DDI-MECHANISM | DDI-EFFECT | DDI-ADVISE | DDI-INT"
  }
]
```

If no valid relation exists in the input text, the output must be `[]`.

## Package Layout

- `configs/`: canonical training and inference configs
- `../MISC/data/`: processed dataset files included in this package
- `../MISC/evaluate_datasets/`: benchmark index references and external dataset notes
- `../MISC/outputs/`: retained LoRA adapter and observability files
- `../MISC/reports/`: reproducibility notes and command history
- `../MISC/results/`: retained summary-level evaluation outputs
- `flash_attn/`: lightweight local compatibility layer for `apply_rotary`
- `models/`: expected local location of the base model
- `prompts/`: shared system prompt
- `scripts/`: train, inference, Gradio, and evaluation entrypoints
- `src/`: core Python modules used by the main pipeline

## Canonical Files

Use the following files as the default reproducible path for this package:

- training config: `configs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml`
- inference config: `configs/infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml`
- training data: `../MISC/data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/merged_chatml_train.jsonl`
- validation data: `../MISC/data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/merged_chatml_validation.jsonl`
- test data: `../MISC/data/processed/Comp6713-Ddi-Ade-Extraction_latest_raw_clean/merged_chatml_test.jsonl`
- retained adapter: `../MISC/outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter/`

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `huggingface_hub` is missing in your environment, install it as well because the
download helper below uses it directly:

```bash
python -m pip install huggingface_hub
```

## Download The Base Model

The package does not bundle `models/Qwen3-8B`. Download it into the expected path with:

```bash
bash scripts/setup/download_base_model.sh
```

Optional arguments:

```bash
bash scripts/setup/download_base_model.sh Qwen/Qwen3-8B /absolute/path/to/models/Qwen3-8B
```

Notes:

- Set `HF_TOKEN` in your shell first if the model requires an authenticated Hugging Face session.
- The default destination is `models/Qwen3-8B`, which matches the packaged Echo configs.
- If you store the model somewhere else, use `--base-model /path/to/Qwen3-8B` for CLI inference,
  or set `COMP6713_BASE_MODEL=/path/to/Qwen3-8B` before launching the Gradio demo.

## Check The Local Base Model

The package does not include the full base model weights. By default both training and inference expect:

```text
models/Qwen3-8B
```

Check whether that path already exists on your machine:

```bash
python -c "from pathlib import Path; p = Path('models/Qwen3-8B'); print(p.resolve()); print('exists =', p.exists())"
```

If the command prints `exists = False`, run `bash scripts/setup/download_base_model.sh`
or override the path with `--base-model`.

## Reproduce The Packaged LoRA Run

Smoke-check the training setup:

```bash
python -B scripts/train/train_finetune.py \
  --config configs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml \
  --dry-run-samples 8
```

Run training:

```bash
python -B scripts/train/train_finetune.py \
  --config configs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml \
  --do-train
```

Run validation inference with the retained adapter:

```bash
python -B scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml
```

Run test-set inference with the retained adapter:

```bash
python -B scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml \
  --split test \
  --output-path ../MISC/results/inference_runs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3_test_predictions.jsonl \
  --metrics-path ../MISC/results/inference_runs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3_test_metrics.txt \
  --metrics-json-path ../MISC/results/inference_runs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3_test_metrics.json
```

Evaluate an existing prediction file:

```bash
python -B scripts/evaluation/evaluate_predictions.py \
  --predictions-path ../MISC/results/inference_runs/your_predictions.jsonl
```

Generate the compact quantitative comparison report:

```bash
python -B scripts/evaluation/quantitative_evaluation.py
```

## Run The Base Model Only

The packaged inference config points to the retained LoRA adapter by default. To ignore it and run the base model only, add `--disable-adapter`.

Validation-set inference with the base model:

```bash
python -B scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml \
  --disable-adapter \
  --output-path ../MISC/results/inference_runs/qwen3_8b_base_validation_predictions.jsonl \
  --metrics-path ../MISC/results/inference_runs/qwen3_8b_base_validation_metrics.txt \
  --metrics-json-path ../MISC/results/inference_runs/qwen3_8b_base_validation_metrics.json
```

Single-text base inference:

```bash
python -B scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml \
  --disable-adapter \
  --input-text "Warfarin may cause bleeding."
```

If the base model is not stored under `models/Qwen3-8B`, append `--base-model /path/to/your/model`.

## Run With The LoRA Adapter

The retained adapter is used automatically by the packaged inference config. The simplest LoRA command is:

```bash
python -B scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml
```

Single-text LoRA inference:

```bash
python -B scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml \
  --input-text "Warfarin may cause bleeding."
```

To point at a different adapter directory explicitly:

```bash
python -B scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml \
  --adapter-path ../MISC/outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter
```

## Launch Gradio

Launch the Gradio UI:

```bash
python -B scripts/gradio/app.py \
  --host 127.0.0.1 \
  --port 7860
```

What the Echo Gradio UI does:

- `Base only (Qwen3-8B)` loads `configs/infer_gradio_base.yaml`
- `+ LoRA (repo adapter)` loads `configs/infer_gradio_balanced_e3.yaml`
- The demo auto-loads the selected preset on page open

If port `7860` is busy, the demo tries the next free port up to `--port-retries`.

Create a temporary Gradio public link:

```bash
python -B scripts/gradio/app.py \
  --host 127.0.0.1 \
  --port 7860 \
  --share
```

If the base model is stored elsewhere, export:

```bash
export COMP6713_BASE_MODEL=/path/to/Qwen3-8B
```

## Git LFS For `*.safetensors`

The retained adapter contains:

```text
../MISC/outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter/adapter_model.safetensors
```

This repository now tracks `*.safetensors` with Git LFS via `Echo/.gitattributes`.

If you clone or pull this package in a fresh environment, run:

```bash
git lfs install
git lfs pull
```

## Included Artifacts

- The processed `latest_raw_clean` dataset is included directly in `../MISC/data/`.
- The retained LoRA `final_adapter/` is included in `../MISC/outputs/`.
- Summary-only benchmark outputs are included in `../MISC/results/`.

Large historical checkpoints and raw benchmark prediction dumps are intentionally omitted to keep the submission package manageable.

## Recommended Reading Order

1. `README.md`
2. `models/README.md`
3. `configs/README.md`
4. `../MISC/data/README.md`
5. `../MISC/outputs/README.md`
6. `../MISC/results/README.md`
7. `../MISC/reports/RUN_RESULTS.md`
