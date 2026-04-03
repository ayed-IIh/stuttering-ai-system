# Training pipeline runbook

Step-by-step guide to go from a **fresh repository checkout** to **trained checkpoints**, **evaluation metrics**, and **inference export**. Canonical config details live in `ai/training/configs/CONFIG_SCHEMA.md`; preprocessing rules in `docs/preprocessing_rules.md`.

**Validation:** This procedure should be run end-to-end on a clean clone by **Ali** (or the assigned owner) and updated if any step drifts from the codebase. See [§9 Sign-off](#9-sign-off-ali--fresh-checkout-validation).

---

## 1. Prerequisites

### 1.1 Python

- Use **Python 3.10** (matches CI in `.github/workflows/ci.yml`). **3.10+** is acceptable if your environment tracks README, but prefer 3.10 for parity with automated checks.

### 1.2 Virtual environment (recommended)

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 1.3 CUDA / PyTorch (GPU)

- **CPU-only:** `pip install -r requirements.txt` installs a CPU build of PyTorch from the index you are using.
- **NVIDIA GPU:** Install a **CUDA-enabled** PyTorch build that matches your driver from [PyTorch — Get Started](https://pytorch.org/get-started/locally/) *before* or *instead* of relying on the default CPU wheel, then install the rest of the stack:

  ```bash
  pip install -r requirements.txt -r requirements-dev.txt
  ```

  If you already installed CPU PyTorch, reinstall the correct `torch` / `torchaudio` pair for your CUDA version per PyTorch’s instructions.

- Training uses `output.device: auto` by default (`cuda` → `mps` → `cpu`). Override with `cuda` or `cpu` in the YAML if needed.

### 1.4 Project dependencies

From the **repository root** (the directory that contains `requirements.txt`):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

### 1.5 Optional: AWS for dataset download

`scripts/download_dataset.py` uses **boto3** and expects credentials (environment variables or `~/.aws/credentials`). Optional `.env` is loaded where scripts use `python-dotenv` (see script headers).

### 1.6 Hugging Face cache

First training run downloads `facebook/wav2vec2-base` (and processor files). Ensure sufficient disk space and network access.

---

## 2. Data preparation (order matters)

These steps assume you work from the **repo root** unless a path is absolute.

### 2.1 Get raw audio under `ai/dataset/raw/`

**Option A — Sync from S3 (typical for a fresh checkout)**

```bash
python scripts/download_dataset.py --version v1.0
```

This mirrors `s3://<bucket>/raw/<version>/…` into canonical folders:

`ai/dataset/raw/<class_name>/*.wav` (class names must match `shared/labels.py`).

**Option B — Local drop**

Copy labeled `.wav` files into the same folder layout under `ai/dataset/raw/`.

### 2.2 Inventory and audit (**Ali / `audit_dataset.py`**)

Builds `dataset_inventory.csv`, human-readable reports, and enforces a minimum valid-file rate:

```bash
python scripts/audit_dataset.py
```

Defaults: dataset root `ai/dataset/raw`, outputs under `ai/dataset/metadata/`. Use `--dataset-root` / `--output-dir` if your layout differs.

- Exit code **0** means ≥95% of discovered files passed validation (see script docstring).
- If many files fail, fix filenames or audio before continuing.

### 2.3 Rename / dedupe (**Ali / `rename_dataset.py`**, if needed)

Uses `dataset_inventory.csv` to preview or apply safe renames and move cross-class duplicates to `_duplicates/`:

```bash
python scripts/rename_dataset.py --dry-run
python scripts/rename_dataset.py --execute
```

Run **after** audit when the report flags naming issues. Re-run **§2.2** after `--execute` so the inventory matches disk.

### 2.4 Audio validation

```bash
python ai/preprocessing/validate_audio.py --inventory-csv ai/dataset/metadata/dataset_inventory.csv
```

Confirms clips against project preprocessing expectations.

### 2.5 Class distribution (**Ali / `class_distribution.py`**)

Recommended before splitting so imbalance is visible:

```bash
python scripts/class_distribution.py
```

Produces plots and `class_distribution_report.md` under `ai/dataset/metadata/`.

### 2.6 Stratified splits (**manifests for training**)

```bash
python ai/dataset/split_dataset.py \
  --inventory-csv ai/dataset/metadata/dataset_inventory.csv \
  --output-dir ai/dataset/processed \
  --seed 42
```

Writes:

- `ai/dataset/processed/train_manifest.csv`
- `ai/dataset/processed/val_manifest.csv`
- `ai/dataset/processed/test_manifest.csv`

Manifests must use path + label columns understood by `ai/training/dataloader.py` (see `CONFIG_SCHEMA.md`).

### 2.7 Curator path only — publishing a new S3 version

If **Ali** is cutting a new dataset version (not required for every training run), follow **`docs/dataset_versioning.md` §5 Promotion Workflow**: `audit_dataset.py` → `class_distribution.py` → `upload_dataset.py` dry-run → `upload_dataset.py` → verify → update the version registry in that doc.

---

## 3. Choosing and editing a training config YAML

1. **Start from a known template**
   - Frozen encoder baseline: `ai/training/configs/baseline_frozen.yaml`
   - Full fine-tune experiment: `ai/training/configs/finetune_full.yaml`

2. **Copy** to a new file (e.g. `ai/training/configs/my_experiment.yaml`) or edit a dedicated experiment file so you do not overwrite shared baselines unintentionally.

3. **Edit with this checklist**
   - **`output.experiment_name`** — unique per run; controls subfolders under `checkpoint_dir` and `log_dir`.
   - **`data.train_manifest` / `val_manifest` / `test_manifest`** — paths relative to **current working directory** when you launch `train.py`, or use absolute paths.
   - **`training`**: `num_epochs`, `batch_size`, `learning_rate`, `num_workers`, etc.
   - **`model`**: `freeze_encoder`, `dropout_rate`, `model_name`.
   - **`output.device`**: `auto`, `cuda`, `mps`, or `cpu`.

4. **Schema reference:** `ai/training/configs/CONFIG_SCHEMA.md`.

---

## 4. Launching training (`train.py`)

From the **repository root** (so default manifest paths resolve):

```bash
python ai/training/train.py --config ai/training/configs/baseline_frozen.yaml
```

Checkpoints and logs are written under:

- `ai/training/checkpoints/<experiment_name>/best_model.pt` — best **validation macro-F1**
- `ai/training/checkpoints/<experiment_name>/last_model.pt` — last epoch
- `ai/training/logs/<experiment_name>/training_log.csv` — per-epoch metrics

Use **Ctrl+C** to interrupt; the trainer saves a last checkpoint before exit when handled cleanly (see `train.py`).

---

## 5. Reading `training_log.csv`

Location:

```text
ai/training/logs/<experiment_name>/training_log.csv
```

Columns (append order, one row per epoch):

| Column           | Meaning |
|------------------|--------|
| `epoch`          | 1-based epoch index |
| `train_loss`     | Mean training cross-entropy for the epoch |
| `val_loss`       | Mean validation cross-entropy |
| `val_accuracy`   | Validation accuracy |
| `val_macro_f1`   | Validation macro-averaged F1 (primary ranking metric for `best_model.pt`) |
| `learning_rate`  | LR at end of epoch (scheduler-dependent) |

**During training:** tail the file or open it in a spreadsheet; `val_macro_f1` should generally trend up over epochs for a healthy run.

**After training:** the best checkpoint corresponds to the epoch with highest `val_macro_f1` (also reflected in checkpoint metadata).

---

## 6. Evaluation (`evaluate.py`) and `metrics.json`

Run **after** you have a checkpoint and a **test** manifest (paths relative to CWD or absolute):

```bash
python ai/evaluation/evaluate.py \
  --checkpoint_path ai/training/checkpoints/<experiment_name>/best_model.pt \
  --manifest_path ai/dataset/processed/test_manifest.csv \
  --output_dir ai/evaluation/runs/<run_name>
```

**Outputs**

- **`metrics.json`** — test-set `accuracy`, `f1_macro`, `f1_weighted`, per-class precision/recall/F1, support counts, confusion matrices, paths to artifacts, and optional `data_warning` for repetition subclasses (see `ai/evaluation/ACCEPTANCE_CRITERIA.md` if present in your branch).
- **`confusion_matrix.png`** — normalized 7×7 heatmap.

**Interpretation**

- Compare **`f1_macro`** and **`accuracy`** to team gates in `ACCEPTANCE_CRITERIA.md` (if committed).
- Inspect **`per_class`** for weak classes; align with `class_distribution_report.md`.
- Treat **`data_warning`** as a data-quality / balance signal, not a training bug by itself.

---

## 7. Export inference artifacts (`checkpoint_utils` export)

Produces a lightweight bundle for backend / deployment handoff (weights + config JSON):

```bash
python -m ai.training.checkpoint_utils export \
  --checkpoint_path ai/training/checkpoints/<experiment_name>/best_model.pt \
  --output_dir path/to/inference_bundle
```

Writes (see `ai/training/checkpoint_utils.py`):

- `model_inference.pt` — `model_state_dict` + `config`
- `config.json` — pretty-printed config

---

## 8. Troubleshooting

### 8.1 CUDA out of memory (OOM)

1. **Lower `training.batch_size`** in your YAML (halve until stable).
2. **Set `output.device: cpu`** only as a last resort for debugging; it is much slower.
3. **Gradient checkpointing** is **not** wired in the current `StutteringClassifier` training loop. Reducing batch size is the first lever; enabling Hugging Face–style gradient checkpointing on the encoder would require a **code change** (discuss with the team before merging).

### 8.2 NaN or exploding loss

1. **Lower `training.learning_rate`** (try 3×–10× smaller).
2. **Check audio quality** — re-run `audit_dataset.py` and `validate_audio.py`; corrupt or silent files can destabilize training.
3. **Normalization** — training data flows through the Wav2Vec2 processor + manifest pipeline; confirm manifests point at the correct files and sample rates are sensible (see `docs/preprocessing_rules.md`).

### 8.3 Class imbalance

1. **Measure** — `scripts/class_distribution.py` and `class_distribution_report.md`.
2. **Mitigations (may require code changes today):** The default `train.py` loop uses **uniform shuffling** and **unweighted** cross-entropy. For heavy imbalance, the team may add **`WeightedRandomSampler`** in the DataLoader and/or **`CrossEntropyLoss(weight=…)`**. Coordinate implementation in `ai/training/` rather than relying on a hidden flag.
3. **Data** — prefer collecting more samples for rare classes (see report thresholds).

### 8.4 Empty `DataLoader` / “empty manifest” errors

1. **Current working directory** — run `train.py` from the repo root unless manifests are absolute.
2. **CSV paths** — manifest `file_path` / `path` entries must exist from the machine running training (avoid another developer’s absolute paths).
3. **Regenerate manifests** — `split_dataset.py` after a fresh `audit_dataset.py`.
4. **Filtered inventory** — `split_dataset.py` drops rows with audit flag columns true; if everything is flagged, manifests can end up empty.

---

## 9. Sign-off — Ali / fresh-checkout validation

| Step | Command / artifact | Pass (✓) |
|------|----------------------|----------|
| 1 | Python 3.10 venv; `pip install -r requirements.txt -r requirements-dev.txt` | |
| 2 | Data present; `python scripts/audit_dataset.py` exit 0 | |
| 3 | `python ai/preprocessing/validate_audio.py …` | |
| 4 | `python scripts/class_distribution.py` | |
| 5 | `python ai/dataset/split_dataset.py …` → three manifests | |
| 6 | `python ai/training/train.py --config …` completes ≥1 epoch | |
| 7 | `training_log.csv` shows expected columns | |
| 8 | `python ai/evaluation/evaluate.py …` → `metrics.json` + `confusion_matrix.png` | |
| 9 | `python -m ai.training.checkpoint_utils export …` → `model_inference.pt` + `config.json` | |

**Owner:** Ali  
**Date validated:** _______________  
**Branch / commit:** _______________  
**Notes (GPU type, dataset version, anomalies):**

---

*If any command or path in this runbook disagrees with the code on your branch, update this document in the same PR as the code change.*
