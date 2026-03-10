# Project Overview

## Project Vision
Build a production-ready AI platform that detects and analyzes stuttering patterns from speech recordings, enabling accurate, explainable, and scalable speech assessment workflows.

## Dataset Structure
The dataset layer should separate data by lifecycle stage:
- `ai/dataset/raw/`: immutable source audio recordings (`.wav`) and source metadata.
- `ai/dataset/interim/`: cleaned or segmented intermediate outputs.
- `ai/dataset/processed/`: model-ready tensors/features and labels.
- `ai/dataset/metadata/`: manifests, label mappings, speaker/session metadata.

Recommended conventions:
- Use stable file IDs and consistent naming (`speakerId_sessionId_clipId.wav`).
- Version dataset manifests and label definitions in Git.
- Keep large audio and binary artifacts outside Git (object storage or DVC in future).

## AI Pipeline
1. Data ingestion: collect `.wav` files and metadata.
2. Preprocessing: resample audio, denoise, trim silence, and normalize.
3. Feature extraction: generate features such as mel spectrograms / embeddings.
4. Training: train stuttering classification/detection models.
5. Evaluation: compute metrics (F1, precision, recall, confusion matrix).
6. Model packaging: export best models for backend inference services.

## System Architecture
- AI modules (`ai/`) handle experiments, training, and model quality.
- Backend modules (`backend/`) provide APIs, inference orchestration, and persistence.
- PostgreSQL stores metadata, prediction logs, and operational records.
- Future AWS deployment can host model inference services and scalable storage.

## Team Responsibilities
- Ali — data ingestion, labeling flow, and preprocessing data pipeline.
- Adan — model architecture design, training strategy, and evaluation quality.
- Saddouq — AI-backend integration for model serving and inference workflows.
- Wael — backend API, database schema, and operational reliability.
