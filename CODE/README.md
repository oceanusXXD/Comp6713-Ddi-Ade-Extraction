# COMP6713 DDI/ADE Extraction

Code, configs, prompts, and reports for ADE and DDI extraction.

## Layout

- `configs/`: training, inference, and demo YAML files
- `data/`: dataset path references used by configs and scripts
- `evaluate_datasets/`: benchmark download and indexing scripts
- `flash_attn/`: local compatibility layer for `apply_rotary`
- `models/`: default base model path reference
- `outputs/`: training and adapter output paths
- `prompts/`: shared system prompt files
- `reports/`: recorded verification checks
- `resources/`: baseline lexicons and helper files
- `results/`: runtime prediction and metric outputs
- `scripts/`: command-line entrypoints
- `src/`: Python source modules

## Primary files

- Train config: `configs/qwen3_8b_lora_ddi_ade_final.yaml`
- Inference config: `configs/infer_qwen3_8b_lora_ddi_ade_final.yaml`
- Train script: `scripts/train/train_finetune.py`
- Inference script: `scripts/inference/predict.py`
- Evaluation script: `scripts/evaluation/evaluate_predictions.py`
- Quantitative summary script: `scripts/evaluation/quantitative_evaluation.py`

## Setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the spaCy English model only if you use `src/baseline/`:

```bash
python -m spacy download en_core_web_sm
```

## Common commands

Rebuild the processed dataset:

```bash
python scripts/analysis/audit_and_prepare_final_dataset.py
```

Run a training dry run:

```bash
python scripts/train/train_finetune.py \
  --config configs/qwen3_8b_lora_ddi_ade_final.yaml \
  --dry-run-samples 8
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

Generate the benchmark comparison summary:

```bash
python scripts/evaluation/quantitative_evaluation.py
```

Start the demo:

```bash
python scripts/demo/gradio_demo.py --host 127.0.0.1 --port 7860
```

## References

- Data paths: `data/README.md`
- Model path: `models/README.md`
- Output paths: `outputs/README.md`
- Results notes: `results/README.md`
- Recorded checks: `reports/RUN_RESULTS.md`
