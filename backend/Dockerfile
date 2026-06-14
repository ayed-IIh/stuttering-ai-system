FROM python:3.10-slim

# Prevent Python from buffering logs / writing .pyc; disable pip cache.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libs for audio decoding (soundfile/librosa need libsndfile; torchaudio
# codecs need ffmpeg). Without these, /predict fails at decode time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the HuggingFace Wav2Vec2 processor AND base model weights into the
# image cache so model load works OFFLINE at runtime. The classifier rebuilds
# its encoder via Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base') before
# loading our fine-tuned weights, so BOTH the processor and the base model must
# be cached — otherwise a cold container reaches HuggingFace Hub and aborts
# startup in PRODUCTION_MODE.
RUN python -c "from transformers import Wav2Vec2Processor, Wav2Vec2Model; Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base'); Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')"

# Copy application code. NOTE: `shared/` is REQUIRED — routes.py and
# model_service.py do `from shared.labels import ...`; omitting it (as the
# previous Dockerfile did) crashes the container at import time.
COPY shared/ /app/shared/
COPY backend/ /app/backend/
COPY ai/ /app/ai/

# Bake the production model artifact (v3.0 single-label, ~360 MB) so the image
# is self-contained and needs no volume mount to serve predictions.
COPY exports/v3.0/ /app/exports/v3.0/

# In-image defaults (override at deploy time via env / compose / .env).
# MODEL_PATH is ABSOLUTE so it resolves regardless of process CWD.
ENV MODEL_PATH=/app/exports/v3.0 \
    MODEL_SOURCE=local \
    DEVICE=cpu \
    FEEDBACK_DIR=/app/feedback_data \
    SERVICE_VERSION=v3.0

# HITL feedback (therapist corrections) must survive container restarts — mount
# a named volume here in production.
VOLUME ["/app/feedback_data"]

EXPOSE 8000

# Production server: no --reload (dev-only), single worker (each worker loads a
# full model copy into RAM). Scale workers only if you have the memory.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
