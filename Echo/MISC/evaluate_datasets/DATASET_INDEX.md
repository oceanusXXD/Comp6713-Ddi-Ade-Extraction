# Evaluation Dataset Index

## Current Internal Train / Validation / Test Data

- Train set: `/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train.jsonl`, total `4924` rows
- Validation set: `/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_validation.jsonl`, total `436` rows
- Test set: `/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_test.jsonl`, total `619` rows
- Augmentation sidecar: `/home/coder/data/Comp6713-Ddi-Ade-Extraction/data/processed/Comp6713-Ddi-Ade-Extraction_final_augment/merged_chatml_train_augmentations.jsonl`, total `1012` rows
- Augmentation type distribution: `{"hardcase": 247, "margincase": 265, "negative": 320, "paraphrase": 180}`

## Current External Evaluation Data

- Same-style held-out validation set: `/home/coder/data/Comp6713-Ddi-Ade-Extraction/evaluate_datasets/seen_style_core/official_held_out/merged_chatml_validation.jsonl`, total `436` rows
- Same-style held-out test set: `/home/coder/data/Comp6713-Ddi-Ade-Extraction/evaluate_datasets/seen_style_core/official_held_out/merged_chatml_test.jsonl`, total `619` rows

## External Bundle Directories

- `ade_transfer`: status `complete`
- `ddi_transfer`: status `complete`
- `general_guardrails`: status `complete`
- `pharmacovigilance_cross_genre`: status `complete`
- `seen_style_core`: status `partial`

## Usage

- Re-download and reorganize the external evaluation datasets: `bash evaluate_datasets/download_evaluate_datasets.sh`
- Rebuild the index: `.venv/bin/python evaluate_datasets/build_dataset_index.py`
