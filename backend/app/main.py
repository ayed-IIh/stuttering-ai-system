"""FastAPI app for backend integration (Phase 2 target)."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Stuttering AI System API", version="0.1.0")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Service health endpoint used by deployments and monitoring."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    """Basic root endpoint for quick service verification."""
    return {"message": "Stuttering AI backend is running"}


# Placeholder endpoint for upcoming model inference integration.
@app.post("/predict")
def predict_placeholder() -> dict[str, str]:
    """Return a placeholder response until model serving is implemented."""
    return {"detail": "Inference endpoint will be implemented in Phase 2."}