# Gradio demo

English UI for direct text inference.

- `Base only (Qwen3-8B)`: frozen base model, no PEFT.
- `+ LoRA (repo adapter)`: same base model plus the repository-relative adapter at `outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter`.

This demo reuses the same lower-level inference backend as `scripts/inference/predict.py`, but it is an interactive path:

- no dataset file is required for free-form input
- Gradio forces the demo onto the local `transformers` path
- examples are optional quick-fill helpers only

## Run (repo root)

```bash
cd /path/to/Comp6713-Ddi-Ade-Extraction
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install gradio
.venv/bin/python scripts/demo/gradio_demo.py --host 127.0.0.1 --port 7860
```

Open the URL printed in the terminal. If the port is busy, pass another `--port` or rely on automatic port probing (`--port-retries`).

## Notes

- The LoRA preset keeps a repo-relative adapter path, so the repo can move across machines without rewriting absolute paths.
- Base path resolution falls back through `../models/Qwen3-8B`, repo-local `models/Qwen3-8B`, then Hugging Face only when `allow_remote_model_source: true` and no local base exists.
- Examples come from `merged_chatml_test.jsonl` (or fallbacks under `data/processed/...`) and are optional.
