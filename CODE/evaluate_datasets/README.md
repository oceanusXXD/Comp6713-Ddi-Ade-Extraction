# Evaluation Datasets

Scripts for downloading and indexing benchmark datasets.

## Files

- `download_evaluate_datasets.sh`: download and organize datasets
- `build_dataset_index.py`: rebuild the dataset index

## Commands

Download datasets:

```bash
bash evaluate_datasets/download_evaluate_datasets.sh
```

Force a fresh download:

```bash
bash evaluate_datasets/download_evaluate_datasets.sh --force
```

Rebuild the index:

```bash
python evaluate_datasets/build_dataset_index.py
```
