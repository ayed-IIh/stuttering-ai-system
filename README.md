# Stuttering AI System

End-to-end stuttering speech classification project with:
- audio data pipeline
- model training pipeline (Wav2Vec2-based classifier)
- backend foundation (FastAPI + PostgreSQL + Docker)

The project predicts 7 classes from `.wav` audio.

## Label Taxonomy
Canonical labels live in `shared/labels.py`:

| Label | ID |
|---|---:|
| fluent | 0 |
| blocks | 1 |
| interjections | 2 |
| prolongations | 3 |
| part_word_repetition | 4 |
| phrase_repetition | 5 |
| word_repetition | 6 |

## Current Implementation Status
Implemented and merged:
- Dataset inventory and anomaly audit outputs under `ai/dataset/metadata/`
- Audio validation script `ai/preprocessing/validate_audio.py`
- Class distribution analysis artifacts and plots
- Stratified split manifests under `ai/dataset/processed/`
- Core audio preprocessing utilities in `ai/preprocessing/audio_loader.py`
- Dataset class and dataloader integration
- Training augmentations in `ai/preprocessing/augmentation.py`
- Wav2Vec2 classifier, training pipeline, config system, checkpoint/export utilities
- Backend health endpoint (`/health`)
- PostgreSQL schema/ORM/CRUD foundation
- Docker local stack and CI workflow
- S3 architecture and IAM policy documentation

Still in progress for backend serving:
- Full inference service and `/api/v1/*` prediction routes
- Backend integration test suite for prediction routes
- Final model evaluation workflow (`ai/evaluation/evaluate.py`)

## Repository Structure
```text
stuttering-ai-system/
|- ai/
|  |- dataset/                      # split logic, manifests, metadata
|  |- preprocessing/                # load/normalize/trim/pad/augment/validate
|  |- models/                       # stuttering classifier architecture
|  |- training/                     # train loop, configs, checkpoints, export
|  `- evaluation/                   # evaluation module (partially pending)
|- backend/
|  |- app/                          # FastAPI app entrypoint
|  |- db/                           # schema, models, CRUD, init
|  |- api/                          # API route package (scaffolding)
|  `- services/                     # service layer (scaffolding)
|- shared/                          # shared taxonomy/constants
|- scripts/                         # dataset upload/download and utility scripts
|- docs/                            # architecture and operational docs
|- tests/                           # unit tests
|- docker-compose.yml               # base compose stack
|- docker-compose.dev.yml           # standalone dev compose stack
|- requirements.txt
|- requirements-dev.txt
`- README.md
```

## Requirements
- Python 3.10+
- Docker Desktop (optional, for containerized local run)

Install dependencies:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Local Development Run
Start backend directly:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Check endpoints:
- `GET /health` -> `{"status":"ok"}`
- `GET /docs` -> Swagger UI

## Docker Run (Standalone Dev Stack)
The `docker-compose.dev.yml` file is standalone and can be run directly:

```bash
docker compose -f docker-compose.dev.yml up --build -d
curl http://127.0.0.1:8000/health
docker compose -f docker-compose.dev.yml down --remove-orphans
```

Services:
- backend: `http://127.0.0.1:8000`
- postgres: `localhost:5432`
- pgadmin: `http://127.0.0.1:5050`

## Data and Training Commands
Validate inventory audio:

```bash
python ai/preprocessing/validate_audio.py --inventory-csv ai/dataset/metadata/dataset_inventory.csv
```

Generate stratified manifests:

```bash
python ai/dataset/split_dataset.py --inventory-csv ai/dataset/metadata/dataset_inventory.csv --output-dir ai/dataset/processed --seed 42
```

Train baseline model:

```bash
python ai/training/train.py --config ai/training/configs/baseline_frozen.yaml
```

Export inference artifacts from checkpoint:

```bash
python -m ai.training.checkpoint_utils export --checkpoint_path <path_to_checkpoint.pt> --output_dir <output_dir>
```

## Quality Checks
Run lint:

```bash
flake8 ai backend --config=.flake8
```

Run tests:

```bash
pytest tests -q
```

## Key Docs
- `docs/environments.md`
- `docs/preprocessing_rules.md`
- `docs/db_schema.md`
- `docs/s3_architecture.md`
- `docs/iam_policies.md`
- `docs/git_workflow.md`
- `docs/dataset_versioning.md`

## Branching
- `main`: stable branch
- `dev`: integration branch
- `feature/*`: task branches
