# Dataset Versioning Policy

Defines how raw dataset versions are tagged, uploaded, and managed in S3.

---

## 1. Overview

The raw labeled `.wav` dataset is immutable once uploaded. Each release gets its own versioned prefix in S3 so training runs can always be traced to an exact data snapshot. Old versions are never deleted — they may be marked deprecated but remain accessible.

Version format: `vMAJOR.MINOR`

---

## 2. Version Registry

| Version | Date       | Description                                              | Curator | File Count |
|---------|------------|----------------------------------------------------------|---------|------------|
| v1.0    | 2026-03-30 | Initial clinical recordings across 7 stuttering classes  | Ali     | TBD        |

> Update file count after running `upload_dataset.py` and noting the grand total from the verification output.

---

## 3. Increment Rules

### MINOR bump (`v1.0` → `v1.1`)

Use when the change is additive and the label schema stays the same:

- New speakers added to one or more existing classes
- Corrected audio quality (re-recorded files for the same speakers)
- Additional recordings that expand an existing class without changing its definition

### MAJOR bump (`v1.x` → `v2.0`)

Use when downstream model code must be updated to consume the new data:

- A new stuttering class is added (requires `shared/labels.py` change)
- Files are removed or reclassified between existing classes
- Label definitions change (e.g. "blocks" is redefined to exclude glottal stops)
- File format changes (e.g. switching from 16 kHz mono to 22 kHz stereo)

---

## 4. S3 Structure

```
s3://stuttering-ai-data/
└── raw/
    └── {version}/          e.g. raw/v1.0/
        ├── fluent/
        ├── blocks/
        ├── interjections/
        ├── prolongations/
        ├── part_word_repetition/
        ├── phrase_repetition/
        └── word_repetition/
```

**Immutability:** Never overwrite or delete objects under an existing version prefix. If a file needs to be corrected, cut a new version and upload the corrected set there.

**Canonical label names** come from `shared/labels.py` (`CLASS_LABELS`). S3 folder names must match those strings exactly — no aliases, no camel case.

---

## 5. Promotion Workflow

Before uploading a new version, run through this checklist:

1. **Audit** — run `python scripts/audit_dataset.py` and confirm exit code 0 (≥95% valid files)
2. **Distribution check** — run `python scripts/class_distribution.py` and review class counts
3. **Dry-run upload** — `python scripts/upload_dataset.py --version vX.Y --dry-run` and confirm the file list looks right
4. **Upload** — `python scripts/upload_dataset.py --version vX.Y`
5. **Verify** — confirm per-class S3 counts in the script output match `ai/dataset/metadata/dataset_inventory.csv`
6. **Register** — add a row to the Version Registry table above and commit this file

---

## 6. Retention Policy

All versions are retained indefinitely. If a version is found to be corrupted or was uploaded in error:

- Do **not** delete the S3 prefix
- Add a `DEPRECATED` note in the Version Registry with the reason
- Upload a corrected version under the next version tag

This keeps training run provenance intact even for bad versions.

---

## 7. Local Development

Download any version to `ai/dataset/raw/` with:

```bash
python scripts/download_dataset.py --version v1.0
```

The script resolves to flat canonical subdirectories:

```
ai/dataset/raw/
├── fluent/
├── blocks/
└── ...
```

This matches the structure expected by downstream tooling that reads the `dataset_inventory.csv` manifest.
