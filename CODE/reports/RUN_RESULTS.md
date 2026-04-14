# Run Results

Recorded checks for the `CODE/` directory.

## CLI help

Inference:

```bash
python -B scripts/inference/predict.py --help
```

```text
usage: predict.py [-h] [--config CONFIG] [--split SPLIT]
                  [--input-path INPUT_PATH] [--input-text INPUT_TEXT]
                  [--backend BACKEND] [--system-prompt SYSTEM_PROMPT]
                  [--output-path OUTPUT_PATH] [--metrics-path METRICS_PATH]
                  [--system-prompt-path SYSTEM_PROMPT_PATH]
                  [--metrics-json-path METRICS_JSON_PATH] [--limit LIMIT]
                  [--batch-size BATCH_SIZE] [--max-new-tokens MAX_NEW_TOKENS]
                  [--temperature TEMPERATURE] [--base-model BASE_MODEL]
                  [--adapter-path ADAPTER_PATH] [--enable-thinking]
                  [--disable-thinking] [--debug]

Run ADE/DDI extraction inference and evaluation.
```

Training:

```bash
python -B scripts/train/train_finetune.py --help
```

```text
usage: train_finetune.py [-h] [--config CONFIG] [--do-train]
                         [--resume-from-checkpoint RESUME_FROM_CHECKPOINT]
                         [--max-train-samples MAX_TRAIN_SAMPLES]
                         [--max-eval-samples MAX_EVAL_SAMPLES]
                         [--dry-run-samples DRY_RUN_SAMPLES]
                         [--enable-thinking] [--disable-thinking]

Fine-tune Qwen-style chat models on ADE/DDI extraction.
```

Evaluation:

```bash
python -B scripts/evaluation/evaluate_predictions.py --help
```

```text
usage: evaluate_predictions.py [-h] --predictions-path PREDICTIONS_PATH
                               [--gold-path GOLD_PATH] [--split SPLIT]
                               [--output-path OUTPUT_PATH]
                               [--json-output-path JSON_OUTPUT_PATH]

Evaluate ADE/DDI prediction jsonl files.
```

## Syntax check

```bash
python -B -m py_compile scripts/train/train_finetune.py scripts/inference/predict.py scripts/evaluation/evaluate_predictions.py src/model_utils.py src/inference_config.py src/runtime_env.py
```

```text
(no output)
```

Result: exit code `0`.

## File checks

Python cache directories:

```powershell
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | ForEach-Object { $_.FullName.Substring($PWD.Path.Length + 1) }
```

```text
(no output)
```

Generated assets:

```powershell
Get-ChildItem -Path . -Recurse -File -Include *.jsonl,*.parquet,*.xml,*.gz,*.zip,*.safetensors,*.bin,*.pt,*.ckpt,*.csv,*.log | ForEach-Object { $_.FullName.Substring($PWD.Path.Length + 1) }
```

```text
(no output)
```

Result: no matching files were present when this report was recorded.
