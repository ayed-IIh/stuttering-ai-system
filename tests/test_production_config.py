from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.config import Settings
from backend.services.model_service import ModelNotLoadedError, ModelService

# Compute repo root for stable file resolution
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_production_requires_local_model_path() -> None:
    with pytest.raises(ValueError, match="MODEL_PATH is required"):
        Settings(
            MODEL_SOURCE="local",
            MODEL_PATH="",
            PRODUCTION_MODE=True,
            allowed_origins_raw="",
        )


def test_production_missing_model_artifacts_fail_startup(tmp_path: Path) -> None:
    settings = Settings(
        MODEL_SOURCE="local",
        MODEL_PATH=str(tmp_path / "missing-model-dir"),
        PRODUCTION_MODE=True,
        allowed_origins_raw="",
    )

    with pytest.raises(ModelNotLoadedError, match="Missing artifact files"):
        ModelService(settings)


def test_env_production_has_required_production_values() -> None:
    body = (REPO_ROOT / ".env.production").read_text(encoding="utf-8")

    assert "PRODUCTION_MODE=true" in body
    assert "LOG_LEVEL=WARNING" in body
    assert "MODEL_PATH=/app/models" in body


def test_dockerfile_runs_two_workers_with_warning_logs() -> None:
    dockerfile = (REPO_ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")

    assert '"--workers", "2"' in dockerfile
    assert '"--log-level", "warning"' in dockerfile
    assert "COPY shared/ /app/shared/" in dockerfile
    assert "--reload" not in dockerfile
