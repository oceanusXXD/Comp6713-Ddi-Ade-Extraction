# Models

The configs in this package expect the base model at:

```text
models/Qwen3-8B
```

The base model weights are not included in the submission package.

Check whether it already exists locally:

```bash
python -c "from pathlib import Path; p = Path('models/Qwen3-8B'); print(p.resolve()); print('exists =', p.exists())"
```

You can either:

- place the model in the path above, you can follow the guide in "CODE\README.md" to download the Qwen3-8B model.
- override it on the CLI with `--base-model /path/to/model`

Base-only inference can be run with:

```bash
python -B scripts/inference/predict.py \
  --config configs/infer_qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml \
  --disable-adapter \
  --base-model /path/to/model
```
