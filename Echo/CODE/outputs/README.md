# Outputs

This folder stores the retained training artifact needed to reproduce inference from the packaged submission.

## Included Output

```text
outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/
```

Included subfolders:

- `final_adapter/`: the retained LoRA adapter used by the packaged inference config
- `observability/`: config snapshots, runtime environment details, training metrics, and dataset statistics

## Retained Adapter

The main packaged adapter lives at:

```text
outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter/
```

Its main weight file is:

```text
outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter/adapter_model.safetensors
```

This repository tracks `*.safetensors` with Git LFS. After cloning, run:

```bash
git lfs install
git lfs pull
```

## Omitted Large Artifacts

The following were intentionally not copied into the submission package:

- intermediate checkpoints
- optimizer state
- scheduler state
- large historical output folders unrelated to the retained adapter

This keeps the package smaller while preserving the exact adapter needed for mainline validation inference.
