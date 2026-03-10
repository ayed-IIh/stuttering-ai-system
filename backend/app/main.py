"""FastAPI entrypoint for Stuttering AI backend."""

from fastapi import FastAPI

app = FastAPI(title="Stuttering AI System API", version="0.1.0")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic health endpoint for service monitoring."""
    return {"status": "ok"}
