from __future__ import annotations

import os
import sys
import wave
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.labels import CLASS_LABELS

# Repo root (parent of ``backend/``)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Allow ``import backend.db.database`` without a live Postgres driver or server.
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DB", "test")


def _stub_async_engine() -> None:
    import sqlalchemy.ext.asyncio as sa_asyncio

    def _fake_create_async_engine(*_a: Any, **_kw: Any) -> MagicMock:
        return MagicMock()

    sa_asyncio.create_async_engine = _fake_create_async_engine  # type: ignore[method-assign]


def _stub_librosa_if_missing() -> None:
    """Stub librosa only when it's actually not installed.

    The original test layout was designed for a slim CI image without librosa.
    With librosa installed, transformers' lazy-import scan trips on a bogus
    stub (no ``__spec__``), so we now only stub when needed.
    """
    import importlib.util
    import sys
    import types

    if importlib.util.find_spec("librosa") is not None:
        return
    effects = types.ModuleType("librosa.effects")
    effects.trim = lambda audio_np, top_db=30.0: (audio_np, None)
    lib = types.ModuleType("librosa")
    lib.effects = effects
    sys.modules.setdefault("librosa", lib)
    sys.modules.setdefault("librosa.effects", effects)


_stub_async_engine()
_stub_librosa_if_missing()


class MockModelService:
    """Mock returning multi-label predictions for any decodable WAV.

    Default behaviour: every class gets a uniform sigmoid score of ``1/n``,
    which is below threshold 0.5, so ``predicted_classes`` is empty. Tests can
    override per-class scores by setting ``mock.next_scores`` to a dict before
    calling predict — useful for happy-path assertions.
    """

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings
        self.next_scores: dict[str, float] | None = None

    async def load(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return True

    def predict(self, audio_bytes: bytes) -> dict[str, Any]:
        from backend.services.model_service import InvalidAudioError

        try:
            with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
                wav_file.readframes(min(64, wav_file.getnframes()))
        except wave.Error as exc:
            raise InvalidAudioError(f"Corrupt or undecodable WAV: {exc}") from exc

        threshold = float(getattr(self._settings, "MULTI_LABEL_THRESHOLD", 0.5))
        if self.next_scores is not None:
            all_scores = {label: float(self.next_scores.get(label, 0.0)) for label in CLASS_LABELS}
        else:
            n = len(CLASS_LABELS)
            all_scores = {label: round(1.0 / n, 6) for label in CLASS_LABELS}
        predicted_classes = sorted(
            [
                {"class": label, "confidence": score}
                for label, score in all_scores.items()
                if score >= threshold
            ],
            key=lambda d: d["confidence"],
            reverse=True,
        )
        return {
            "predicted_classes": predicted_classes,
            "all_scores": all_scores,
            "threshold": threshold,
            "processing_time_ms": 1,
            "model_version": "mock-v1",
        }

    def shutdown(self) -> None:
        return None


@pytest.fixture
def test_settings() -> Any:
    from backend.app.config import Settings, clear_settings_cache

    clear_settings_cache()
    return Settings(
        MODEL_PATH="",
        MODEL_SOURCE="local",
        DEVICE="cpu",
        MAX_AUDIO_SIZE_MB=1,
        allowed_origins_raw="http://localhost:3000",
        PRODUCTION_MODE=False,
        DB_URL="postgresql://test:test@127.0.0.1:5432/test",
        SERVICE_VERSION="mock-suite",
        LOG_LEVEL="WARNING",
    )


@pytest.fixture
def test_app(test_settings: Any) -> FastAPI:
    from backend.api.routes import health, router
    from backend.app.middleware import RequestLoggingMiddleware, register_exception_handlers
    from backend.db.database import get_db

    app = FastAPI(title="Stuttering AI API (integration tests)")
    app.state.settings = test_settings
    app.state.model_service = MockModelService(test_settings)
    app.state.service_version = test_settings.SERVICE_VERSION

    app.add_middleware(RequestLoggingMiddleware)
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.add_api_route("/health", health, methods=["GET"], include_in_schema=True)

    async def _override_get_db() -> Any:
        session = MagicMock()
        res = MagicMock()
        res.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=res)
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    with TestClient(test_app) as tc:
        yield tc


@pytest.fixture
def fixture_wav_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "silence_1s_16k_mono.wav"
