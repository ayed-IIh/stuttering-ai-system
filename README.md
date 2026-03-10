# Stuttering AI System

An AI-powered system for detecting and analyzing stuttering patterns from speech recordings using machine learning.

## Current Phase
- Phase 1 - AI model training (active)
- Phase 2 - Backend API integration (planned)
- Phase 3 - Mobile app integration (planned)

## Project Overview
This repository provides a clean, scalable foundation for training and serving an audio classification model that predicts stuttering type from `.wav` recordings.

The classification labels are:
- Fluent
- Blocks
- Prolongations
- Repetitions
- Interjections

## Tech Stack
- Python
- PyTorch
- Torchaudio
- HuggingFace Transformers
- FastAPI
- PostgreSQL
- AWS (S3 / EC2 / RDS later)

## Project Goal
Build an AI system that analyzes speech recordings and classifies different stuttering types.

## AI Architecture
Pipeline:
1. Audio (`.wav`)
2. Audio preprocessing
3. Feature extraction
4. Speech encoder (`Wav2Vec2` or `HuBERT`)
5. Embedding representation
6. Pooling layer
7. Classifier head
8. Stuttering class prediction

Model structure (Phase 1 baseline):
`Audio -> Wav2Vec2 encoder -> pooling -> classification head -> softmax`

## Dataset Description
The dataset contains labeled `.wav` files for supervised audio classification.

Suggested organization under `ai/dataset/`:
- `raw/` for source recordings
- `interim/` for cleaned/segmented intermediate audio
- `processed/` for model-ready features and manifests
- `metadata/` for label maps and split files

## Repository Structure
```text
stuttering-ai-system/
|- ai/                              # AI workflows, model code, and evaluation logic
|  |- dataset/                      # Dataset organization, manifests, splits, and metadata
|  |- preprocessing/                # Audio loading, normalization, and feature prep utilities
|  |- training/                     # Training entrypoints, loop orchestration, experiment configs
|  |- models/                       # Model architectures (Wav2Vec2/HuBERT classifier variants)
|  `- evaluation/                   # Metrics, reports, confusion matrix, validation scripts
|- backend/                         # API and service layer for model serving/integration
|  |- app/                          # FastAPI app initialization and application settings
|  |- api/                          # Route definitions and request/response schemas
|  |- services/                     # Inference/business logic layer
|  `- db/                           # PostgreSQL access, schema, and persistence utilities
|- scripts/                         # Local automation scripts (data checks, run helpers)
|- docs/                            # Technical documentation and team guidelines
|- requirements.txt                 # Python dependency list
|- .gitignore                       # Ignore rules for datasets, checkpoints, logs, caches
`- README.md                        # Project entry documentation
```

## Team Roles
- Ali - Dataset preparation and dataset pipeline
  Focus: dataset validation, train/validation split, dataset structure.
- Adan - AI model development
  Focus: training pipeline, model experiments, evaluation.
- Saddouq - AI backend integration
  Focus: FastAPI backend, inference endpoint, API integration.
- Wael - Backend and database
  Focus: database schema, API endpoints, data pipeline.

## Development Workflow
1. Create branch from `dev` using `feature/*` naming.
2. Implement scoped changes and run checks locally.
3. Open pull request to `dev`.
4. Peer review and requested fixes.
5. Merge to `dev`, then promote to `main` on release.

## Branch Strategy
- `main`: stable production-ready branch.
- `dev`: integration branch for reviewed features.
- `feature/*`: short-lived task branches.

## Task Management
Shared sheet columns:
- Backlog
- Ready
- In Progress
- Review
- Done
- Blocked