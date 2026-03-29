# Experiment Tracker

This table tracks every planned and completed training run.
Update the row after each run with the final validation macro-F1 and any notes.

| experiment_name | config_file | status  | notes | val_f1_result |
|-----------------|-------------|---------|-------|---------------|
| baseline_frozen | `configs/baseline_frozen.yaml` | planned | Frozen encoder, head-only training at lr=1e-3, batch 16, no augmentation. Fast baseline to validate pipeline. | — |
| finetune_full   | `configs/finetune_full.yaml`   | planned | Full encoder fine-tune at lr=2e-5 with warmup 0.1, batch 8, augmentation on. Expect higher F1 but longer training. | — |

## Status key

| Value       | Meaning |
|-------------|---------|
| planned     | Config created, not yet launched. |
| running     | Training in progress. |
| completed   | Training finished normally. |
| failed      | Run crashed or was aborted — see notes. |
| superseded  | Replaced by a newer experiment. |
