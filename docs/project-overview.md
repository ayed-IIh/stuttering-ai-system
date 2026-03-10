# Project Overview

## Vision
Create a reliable AI system that detects stuttering patterns from speech recordings and serves as the ML core for a future mobile application and therapist dashboard.

## Project Phases
- Phase 1: AI model training (current)
- Phase 2: Backend API integration
- Phase 3: Mobile app integration

## Problem Type
- Task: Audio classification
- Input: `.wav` speech recording
- Output: Predicted stuttering category

Target classes:
- Fluent
- Blocks
- Prolongations
- Repetitions
- Interjections

## Dataset Structure
Inside `ai/dataset/` maintain a clear lifecycle:
- `raw/`: original labeled `.wav` files
- `interim/`: cleaned/chunked files generated during preprocessing
- `processed/`: model-ready features and split manifests
- `metadata/`: label maps, speaker/session information, QA notes

Recommended dataset checks:
- Validate sample rate consistency.
- Detect empty/corrupted files.
- Verify class balance.
- Freeze train/validation splits for reproducibility.

## AI Pipeline
1. Load and validate audio files.
2. Apply preprocessing (resample, normalize, optional trimming).
3. Extract model inputs for encoder.
4. Encode speech using `facebook/wav2vec2-base` or HuBERT.
5. Apply pooling over time dimension.
6. Predict class logits via classification head.
7. Convert logits to probabilities and class labels.

## System Architecture
- AI layer (`ai/`): preprocessing, model training, and evaluation.
- Backend layer (`backend/`): FastAPI inference service and integrations.
- Data layer (`backend/db/`): PostgreSQL for metadata, predictions, and operations.
- Cloud layer (future AWS): storage, deployment, and scaling.

## Team Responsibilities
- Ali: dataset validation, train/validation split, dataset structure.
- Adan: model implementation, training loop, experiment tracking, evaluation.
- Saddouq: inference endpoint, model loading, API integration.
- Wael: database schema, backend endpoints, data flow reliability.

## Workflow and Collaboration
- Branches: `main`, `dev`, `feature/*`
- PR flow: feature branch -> `dev` -> `main`
- Task board columns: Backlog, Ready, In Progress, Review, Done, Blocked