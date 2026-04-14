# Quantitative Evaluation

Comparison of **Base model** (Qwen3-8B, no fine-tuning) vs. **LoRA model** (Qwen3-8B + rsLoRA, epoch-3, step 993).

Metric: **Micro Precision / Recall / F1** — triple-level exact match (`relation_type` + `head_entity` + `tail_entity`, case-insensitive).

## Results

| Dataset | Type | Base P | Base R | Base F1 | LoRA P | LoRA R | LoRA F1 | ΔF1 |
|---------|------|-------:|-------:|--------:|-------:|-------:|--------:|----:|
| Own Validation | Internal |  22.1% |  11.1% |  14.8% |  76.5% |  62.1% |  68.5% | +53.7% |
| Own Test | Internal |  25.3% |  16.6% |  20.1% |  69.5% |  61.3% |  65.2% | +45.1% |
| Seen-Style Val | Internal |  23.2% |  11.2% |  15.1% |  77.2% |  61.9% |  68.7% | +53.6% |
| Seen-Style Test | Internal |  26.6% |  17.4% |  21.1% |  69.7% |  60.8% |  65.0% | +43.9% |
| ADE Corpus v2 | External |  65.2% |  19.6% |  30.1% |  90.5% |  88.0% |  89.2% | +59.1% |
| PHEE Dev | External |  25.2% |  11.3% |  15.6% |  43.9% |  57.2% |  49.7% | +34.1% |
| PHEE Test | External |  26.5% |  12.8% |  17.2% |  41.4% |  54.9% |  47.2% | +30.0% |
| DDI-2013 Test | External |  27.6% |  36.0% |  31.3% |  24.7% |  56.2% |  34.3% | +3.1% |
| TAC-2017 ADR Gold | External |  43.4% |   9.2% |  15.2% |  39.1% |  10.4% |  16.4% | +1.2% |
| CADEC MedDRA | External |  31.9% |   8.2% |  13.1% |  28.1% |  18.2% |  22.1% | +9.0% |

## Summary

| Split | Base Avg F1 | LoRA Avg F1 | Gain |
|-------|------------:|------------:|-----:|
| Internal (4 datasets) | 17.8% | 66.8% | +49.1% |
| External (6 datasets) | 20.4% | 43.1% | +22.7% |

## Notes

- **External datasets** (ADE Corpus v2, PHEE, DDI-2013, TAC-2017 ADR, CADEC) were never seen during training.
- **DDI-2013**: LoRA recall improves substantially but precision drops, indicating the model over-predicts relations.
- **TAC-2017 / CADEC**: harder domain transfer; LoRA shows improved recall over the base model.
- The base model occasionally fails to output valid JSON (parse_success_rate < 1.0 on some datasets).
