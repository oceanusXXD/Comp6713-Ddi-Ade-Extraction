# Scripts

This package keeps only the main command-line entrypoints needed for the core reproducibility path.

## Included Entry Points

- `train/train_finetune.py`: training entrypoint
- `inference/predict.py`: batch and single-text inference entrypoint
- `gradio/app.py`: single-text Gradio demo entrypoint
- `evaluation/evaluate_predictions.py`: prediction-file evaluation entrypoint
- `evaluation/quantitative_evaluation.py`: summary-level comparison report generation

## Suggested Order

1. run a dry training check
2. run training if needed
3. run validation or test inference
4. optionally launch the Gradio demo for interactive single-text checks
5. evaluate the generated prediction file
6. regenerate the quantitative report if benchmark summary files are updated
