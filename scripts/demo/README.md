# Gradio demo

English UI. **Base-only** preset is listed first: frozen causal LM, no PEFT. **`+ LoRA:*`** presets load the same base then attach the saved adapter on top (`PeftModel`), same path as `scripts/inference/predict.py`.

## Run (repo root)

```bash
conda activate comp6713
cd /path/to/Comp6713-Ddi-Ade-Extraction
pip install -r requirements.txt
python scripts/demo/gradio_demo.py --host 127.0.0.1 --port 7860
```

Open the URL printed in the terminal. If the port is busy, pass another `--port` or rely on automatic port probing (`--port-retries`).

## Notes

- Examples come from `merged_chatml_test.jsonl` (or fallbacks under `data/processed/...`).
- Base path: see `configs/infer_gradio_base.yaml` and `src/model_utils.try_resolve_existing_path`.
