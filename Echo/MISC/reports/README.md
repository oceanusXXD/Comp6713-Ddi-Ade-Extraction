# Reports

This folder contains reproducibility notes for the packaged submission.

## Files

- `train_command.txt`: retained command history from the original training workspace
- `training_config_snapshot.json`: saved training config from the retained run
- `train_dataset_stats.json`: saved training dataset statistics
- `validation_dataset_stats.json`: saved validation dataset statistics
- `TRAINING_SUMMARY.md`: concise human-readable summary of the retained run
- `RUN_RESULTS.md`: verification notes for the cleaned `Echo/CODE` package

## Notes

- Historical snapshot JSON files in this folder retain absolute paths from the original training workspace.
- The current packaged runtime paths are defined by the YAML configs under `Echo/CODE/configs/`, not by these retained snapshot files.
