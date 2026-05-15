from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import router
from backend.app.config import Settings
from backend.app.middleware import RequestLoggingMiddleware, register_exception_handlers


class _ModelOK:
    def is_loaded(self) -> bool:
        return True


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)
    register_exception_handlers(app)
    app.include_router(router, prefix="/api/v1")
    app.state.model_service = _ModelOK()
    app.state.db_service = None
    app.state.service_version = "0.1.0-test"
    app.state.settings = Settings(DB_URL="postgresql://x")
    return app


def _payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "audio_base64": base64.b64encode(b"RIFF-test-audio").decode("ascii"),
        "correct_labels": ["blocks"],
        "original_prediction": "fluent",
        "model_version": "model-v1",
    }
    payload.update(overrides)
    return payload


def test_feedback_valid_request_returns_quickly_and_saves_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    with TestClient(app) as client:
        started_at = time.perf_counter()
        response = client.post("/api/v1/feedback", json=_payload())
        elapsed_ms = (time.perf_counter() - started_at) * 1000

    assert response.status_code == 200
    assert elapsed_ms < 100
    assert response.json() == {"status": "accepted"}

    label_dir = tmp_path / "feedback_samples" / "blocks"
    audio_files = list(label_dir.glob("*.wav"))
    metadata_files = list(label_dir.glob("*.json"))
    assert len(audio_files) == 1
    assert len(metadata_files) == 1
    assert audio_files[0].read_bytes() == b"RIFF-test-audio"

    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["timestamp"]
    assert metadata["original_prediction"] == "fluent"
    assert metadata["correct_labels"] == ["blocks"]
    assert metadata["model_version"] == "model-v1"


def test_feedback_saves_copy_for_each_correct_label(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = _make_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/feedback",
            json=_payload(correct_labels=["blocks", "word_repetition"]),
        )

    assert response.status_code == 200
    assert list((tmp_path / "feedback_samples" / "blocks").glob("*.wav"))
    assert list((tmp_path / "feedback_samples" / "word_repetition").glob("*.wav"))


def test_feedback_invalid_label_returns_422() -> None:
    app = _make_app()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/feedback",
            json=_payload(correct_labels=["NotALabel"]),
        )

    assert response.status_code == 422
    assert "Valid labels are" in response.text


def test_feedback_empty_correct_labels_returns_422() -> None:
    app = _make_app()

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=_payload(correct_labels=[]))

    assert response.status_code == 422


def test_feedback_malformed_base64_returns_400() -> None:
    app = _make_app()

    with TestClient(app) as client:
        response = client.post("/api/v1/feedback", json=_payload(audio_base64="!!!!"))

    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_REQUEST"
    assert "base64" in response.json()["message"]
