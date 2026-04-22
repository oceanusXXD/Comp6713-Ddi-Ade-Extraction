# Training Summary

This summary describes the retained mainline run bundled with the submission package.

## Main Run

- config: `configs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3.yaml`
- base model: `Qwen3-8B`
- fine-tuning method: `LoRA` with `rsLoRA`
- target modules: `all-linear`
- adapter output: `outputs/qwen3_8b_lora_ddi_ade_latest_raw_clean_balanced_e3/final_adapter`

## Hyperparameters

- `lora_r = 16`
- `lora_alpha = 32`
- `lora_dropout = 0.08`
- `per_device_train_batch_size = 4`
- `gradient_accumulation_steps = 4`
- `learning_rate = 2.8e-5`
- `num_train_epochs = 3`
- `max_seq_length = 4096`

## Dataset Statistics

Training split:

- rows: `5287`
- normalized relation total: `18718`
- examples over max length: `0`

Validation split:

- rows: `588`
- normalized relation total: `1958`
- examples over max length: `0`

## Training Outcome

From the retained `trainer_state_summary.json`:

- final step: `993`
- final epoch: `3.0`
- final eval loss: `0.3778`
- train loss: `1.6571`

## Purpose In This Package

This summary exists to connect the included config, data, adapter, and quantitative evaluation files into one reproducible story for the submission reader.
