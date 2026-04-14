# Prevalidation

Older zero-shot and quick-check utilities.

## Files

- `run_pretest_hf.py`: Hugging Face backend
- `run_pretest_vllm.py`: vLLM backend
- `summarize_pretest.py`: summary metrics and error review
- `preview_chatml.py`: ChatML preview tool
- `prompt.txt`: prompt used by these scripts

## Commands

Run prevalidation with vLLM:

```bash
python src/prevalidation/run_pretest_vllm.py \
  --model_name Qwen/Qwen3-8B \
  --output_path results/prevalidation/pretest_preds.jsonl \
  --limit 50
```

Summarize results:

```bash
python src/prevalidation/summarize_pretest.py \
  --pred_path results/prevalidation/pretest_preds.jsonl
```
