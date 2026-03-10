# Stuttering AI System

An AI-powered system for detecting and analyzing stuttering patterns from speech recordings using machine learning.

## Project Overview
Stuttering AI System is a collaborative repository for building an end-to-end pipeline that:
- Ingests and organizes speech recordings
- Preprocesses `.wav` audio into ML-ready features
- Trains deep learning models for stuttering pattern detection
- Exposes prediction services through a backend API
- Supports future cloud deployment and production scaling

## Tech Stack
- Python
- PyTorch
- Torchaudio
- FastAPI
- PostgreSQL
- AWS (future deployment)

## Project Goal
Detect different types of stuttering from speech audio (`.wav` files).

## Repository Structure
```text
stuttering-ai-system/
|- ai/                              # AI workflows, training logic, and model artifacts
|  |- dataset/                      # Raw/interim/processed dataset references and metadata
|  |- preprocessing/                # Audio loading, cleaning, segmentation, and feature extraction
|  |- training/                     # Training scripts, experiment runners, and training utilities
|  |- models/                       # Model definitions, checkpoints metadata, and export logic
|  `- evaluation/                   # Metrics, validation scripts, and benchmarking outputs
|- backend/                         # Backend services for inference and integrations
|  |- app/                          # FastAPI application entrypoint and app configuration
|  |- api/                          # API route modules and request/response schemas
|  |- services/                     # Business logic for inference and orchestration
|  `- db/                           # Database access layer, migrations, and persistence logic
|- scripts/                         # Utility scripts for automation, setup, and maintenance tasks
|- docs/                            # Project documentation and technical references
|- requirements.txt                 # Python dependencies for local development
|- .gitignore                       # Ignore rules for artifacts, data, logs, and caches
`- README.md                        # Main repository documentation
```

## System Architecture
The system is split into two primary layers:
- AI Layer (`ai/`): handles dataset preparation, preprocessing, model training, and evaluation.
- Backend Layer (`backend/`): serves trained model inference via API endpoints and manages persistence.

Typical data flow:
1. Audio files are collected and organized under dataset conventions.
2. Preprocessing scripts standardize audio and generate features.
3. Training pipelines consume features to train classification/detection models.
4. Evaluated models are versioned and integrated into backend inference services.
5. FastAPI endpoints expose predictions for client applications.

## Team Roles
- Ali - Data pipeline
- Adan - AI model development
- Saddouq - AI backend
- Wael - Backend & database

## Basic Development Workflow
1. Create a feature branch from `main`.
2. Add or update code in the relevant module (`ai/` or `backend/`).
3. Run local checks and tests before committing.
4. Open a pull request with clear scope and change summary.
5. Perform peer review and merge after approval.

## Suggested GitHub Repository Topics
- ai
- machine-learning
- speech-processing
- audio-classification
- pytorch
- speech-analysis