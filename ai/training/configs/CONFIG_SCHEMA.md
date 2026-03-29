# Experiment Configuration Schema

All training experiments are driven by a single YAML file passed via
`--config` to `ai/training/train.py`.  This document is the authoritative
reference for every field.

---

## Top-level sections

| Section      | Purpose                                          |
|--------------|--------------------------------------------------|
| **model**    | Architecture and pre-trained backbone settings   |
| **training** | Optimiser, scheduler, and loop hyper-parameters  |
| **data**     | Manifest paths, audio constraints, augmentation  |
| **output**   | Experiment identity, checkpoint / log directories, device |

---

## `model`

| Key              | Type    | Required | Default                    | Description |
|------------------|---------|----------|----------------------------|-------------|
| `model_name`     | string  | yes      | `facebook/wav2vec2-base`   | HuggingFace model identifier for the Wav2Vec2 backbone. |
| `num_classes`    | int     | yes      | `7`                        | Number of output classes (must match label set). |
| `freeze_encoder` | bool    | yes      | `true`                     | If `true`, all Wav2Vec2 encoder parameters are frozen; only the classification head trains. |
| `dropout_rate`   | float   | no       | `0.1`                      | Dropout probability applied before the linear classification head. |

**Example:**

```yaml
model:
  model_name: facebook/wav2vec2-base
  num_classes: 7
  freeze_encoder: true
  dropout_rate: 0.1
```

---

## `training`

| Key                 | Type   | Required | Default   | Description |
|---------------------|--------|----------|-----------|-------------|
| `num_epochs`        | int    | yes      | —         | Total training epochs. |
| `batch_size`        | int    | yes      | —         | Samples per mini-batch. |
| `learning_rate`     | float  | yes      | —         | Peak learning rate for AdamW. |
| `warmup_ratio`      | float  | no       | `0.0`     | Fraction of total training steps used for linear warmup. |
| `weight_decay`      | float  | no       | `0.01`    | AdamW weight-decay coefficient. |
| `gradient_clip_norm`| float  | no       | `1.0`     | Max L2 norm for gradient clipping. |
| `scheduler_type`    | string | no       | `linear`  | LR scheduler type. Supported: `linear` (linear warmup then linear decay), `cosine`. |
| `seed`              | int    | no       | `42`      | Global random seed for reproducibility. |
| `num_workers`       | int    | no       | `0`       | DataLoader worker processes. |

**Example:**

```yaml
training:
  num_epochs: 10
  batch_size: 16
  learning_rate: 1.0e-3
  warmup_ratio: 0.0
  weight_decay: 0.01
  gradient_clip_norm: 1.0
  scheduler_type: linear
  seed: 42
  num_workers: 0
```

---

## `data`

| Key                    | Type   | Required | Default | Description |
|------------------------|--------|----------|---------|-------------|
| `train_manifest`       | string | yes      | —       | Path to training CSV manifest (relative to CWD or absolute). |
| `val_manifest`         | string | yes      | —       | Path to validation CSV manifest. |
| `test_manifest`        | string | no       | `null`  | Path to held-out test CSV manifest (used for final evaluation only). |
| `max_duration_sec`     | float  | no       | `null`  | If set, clips longer than this value (seconds) are skipped during loading. |
| `augmentation_enabled` | bool   | no       | `false` | Enable on-the-fly data augmentation (noise injection, time stretch, etc.). |
| `target_sr`            | int    | no       | `16000` | Target sample rate in Hz. Audio at different rates is resampled automatically. |

Manifest CSVs must contain a **path column** (one of `path`, `file_path`,
`absolute_path`) and a **label column** (one of `label`, `class_label`,
`class`).  See `ai/training/dataloader.py` for column-resolution rules.

**Example:**

```yaml
data:
  train_manifest: ai/dataset/processed/train_manifest.csv
  val_manifest:   ai/dataset/processed/val_manifest.csv
  test_manifest:  ai/dataset/processed/test_manifest.csv
  max_duration_sec: 10.0
  augmentation_enabled: false
  target_sr: 16000
```

---

## `output`

| Key               | Type   | Required | Default                      | Description |
|-------------------|--------|----------|------------------------------|-------------|
| `experiment_name` | string | yes      | —                            | Unique name used as a subdirectory under `checkpoint_dir` and `log_dir`. |
| `checkpoint_dir`  | string | no       | `ai/training/checkpoints`    | Root directory for saved model checkpoints. |
| `log_dir`         | string | no       | `ai/training/logs`           | Root directory for training CSV logs. |
| `device`          | string | no       | `auto`                       | Compute device: `auto` (cuda > mps > cpu), `cuda`, `mps`, or `cpu`. |

**Example:**

```yaml
output:
  experiment_name: baseline_frozen
  checkpoint_dir: ai/training/checkpoints
  log_dir: ai/training/logs
  device: auto
```

---

## Full minimal config

```yaml
model:
  model_name: facebook/wav2vec2-base
  num_classes: 7
  freeze_encoder: true

training:
  num_epochs: 10
  batch_size: 16
  learning_rate: 1.0e-3

data:
  train_manifest: ai/dataset/processed/train_manifest.csv
  val_manifest:   ai/dataset/processed/val_manifest.csv

output:
  experiment_name: my_experiment
```

All omitted optional fields fall back to their documented defaults.

---

## Notes

- **Paths** in `data` and `output` are resolved relative to the process
  working directory (typically the repository root).  Absolute paths are
  also supported.
- **Checkpoints** are saved as `<checkpoint_dir>/<experiment_name>/best_model.pt`
  and `last_model.pt`.
- **Logs** are written to `<log_dir>/<experiment_name>/training_log.csv`.
- **scheduler_type** `cosine` uses
  `transformers.get_cosine_schedule_with_warmup`; `linear` uses
  `transformers.get_linear_schedule_with_warmup`.
