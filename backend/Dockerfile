FROM python:3.10-slim

# Prevent Python from buffering logs / writing .pyc; disable pip cache.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Which exported model to bake into the image. Bump in ONE place to ship a new
# version — it drives the COPY path, MODEL_PATH, and SERVICE_VERSION below.
ARG MODEL_VERSION=v3.0

WORKDIR /app

# System libs for audio decoding (soundfile/librosa need libsndfile; torchaudio
# codecs need ffmpeg). Without these, /predict fails at decode time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the unprivileged runtime user up front so the model + cache can be
# COPY --chown'd to it directly — avoids a costly `chown -R` layer that would
# duplicate the ~360 MB model and HF cache in the image.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /opt/hf_cache \
    && chown appuser:appuser /opt/hf_cache /app

USER appuser

# HF cache lives at a shared path (not /root) so the non-root runtime user can
# read it. Pre-bake the Wav2Vec2 processor AND base model weights here (as
# appuser) so model load works OFFLINE at runtime: the classifier rebuilds its
# encoder via Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base') before
# loading our fine-tuned weights, so BOTH must be cached — otherwise a cold
# container reaches HuggingFace Hub and aborts startup in PRODUCTION_MODE.
# NOTE: 'facebook/wav2vec2-base' must match ModelConfig.model_name in the model
# config; if a future export changes the base encoder, update it here too.
ENV HF_HOME=/opt/hf_cache
RUN python -c "from transformers import Wav2Vec2Processor, Wav2Vec2Model; Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base'); Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')"

# Copy application code owned by appuser. `shared/` is REQUIRED — routes.py and
# model_service.py do `from shared.labels import ...`; omitting it crashes the
# container at import time.
COPY --chown=appuser:appuser shared/ /app/shared/
COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser ai/ /app/ai/

# Bake the production model artifact (single-label, ~360 MB) so the image is
# self-contained and needs no volume mount to serve predictions.
COPY --chown=appuser:appuser exports/${MODEL_VERSION}/ /app/exports/${MODEL_VERSION}/

# Feedback dir (a fresh named volume mounted here inherits appuser ownership, so
# the non-root process can still write therapist corrections).
RUN mkdir -p /app/feedback_data

# In-image defaults (override at deploy time via env / compose / .env).
# MODEL_PATH is ABSOLUTE so it resolves regardless of process CWD.
ENV MODEL_PATH=/app/exports/${MODEL_VERSION} \
    MODEL_SOURCE=local \
    DEVICE=cpu \
    FEEDBACK_DIR=/app/feedback_data \
    SERVICE_VERSION=${MODEL_VERSION}

# HITL feedback (therapist corrections) must survive container restarts — mount
# a named volume here in production.
VOLUME ["/app/feedback_data"]

EXPOSE 8000

# Production server: no --reload (dev-only), single worker (each worker loads a
# full model copy into RAM). Scale workers only if you have the memory.
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
