# Gradio Demo

Interactive text inference UI built on `scripts/inference/predict.py`.

## Presets

- `Base only (Qwen3-8B)`
- `+ LoRA (repo adapter)`: `outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter`

## Run

```bash
python -m pip install -r requirements.txt
python scripts/demo/gradio_demo.py --host 127.0.0.1 --port 7860
```

## Notes

- Free-form input does not require a dataset file.
- The demo uses the local `transformers` backend.
- Example rows are optional.
