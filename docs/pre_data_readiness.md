# Pre-Data Readiness

This checklist is for the period before client data arrives.

Goal:
- keep the project in a deployable and train-ready state
- run smoke/integration checks regularly
- avoid last-minute breakage when final data is delivered

## One Command Readiness Check

Run from repo root:

```bash
python scripts/pre_data_readiness.py
```

It performs:
- required file checks across AI and backend layers
- canonical 7-label taxonomy sanity checks
- training config schema checks
- manifest schema and label consistency checks
- lint (`flake8 ai backend --config=.flake8`)
- unit tests (`pytest tests -q`)
- backend integration tests (`pytest backend/tests -q`)
- model smoke test (`python scripts/test_classifier_smoke.py`)

It writes a report to:

```text
ai/training/logs/pre_data_readiness_report.json
```

## Optional Flags

```bash
python scripts/pre_data_readiness.py --skip-lint
python scripts/pre_data_readiness.py --skip-tests
python scripts/pre_data_readiness.py --skip-smoke
python scripts/pre_data_readiness.py --report-path ai/training/logs/readiness_custom.json
```

## Immediate Steps Once Client Data Arrives

1. Sync/download client dataset to agreed path.
2. Rebuild inventory:
   - `python scripts/audit_dataset.py ...`
3. Validate audio quality:
   - `python ai/preprocessing/validate_audio.py --inventory-csv ai/dataset/metadata/dataset_inventory.csv`
4. Re-split dataset:
   - `python ai/dataset/split_dataset.py --inventory-csv ai/dataset/metadata/dataset_inventory.csv --output-dir ai/dataset/processed --seed 42`
5. Re-run readiness check:
   - `python scripts/pre_data_readiness.py`
6. Start final training:
   - `python ai/training/train.py --config ai/training/configs/baseline_frozen.yaml`
7. Run evaluation:
   - `python ai/evaluation/evaluate.py --checkpoint_path ... --manifest_path ai/dataset/processed/test_manifest.csv --output_dir ...`

