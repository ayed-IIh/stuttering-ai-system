from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import boto3
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from moto import mock_aws

from backend.services.feedback_service import FeedbackStore
from backend.services.s3_storage import S3Storage
from backend.tests.conftest import MockModelService

BUCKET = "stt-s3-itest"


@pytest.fixture
def s3_app(test_settings: Any, fixture_wav_path: Path):
    """An app wired to a moto-mocked S3 for both /predict and /feedback."""
    from backend.api.routes import health, router
    from backend.app.middleware import (
        RequestLoggingMiddleware,
        register_exception_handlers,
    )
    from backend.db.database import get_db

    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        s3 = S3Storage(BUCKET, region="us-east-1")
        s3.upload_bytes(
            "predict-audio/clip1.wav", fixture_wav_path.read_bytes(), "audio/wav"
        )

        app = FastAPI(title="S3 integration test")
        app.state.settings = test_settings
        app.state.model_service = MockModelService(test_settings)
        app.state.service_version = "s3-itest"
        app.state.s3 = s3
        app.state.s3_predict_prefix = "predict-audio/"
        app.state.feedback_store = FeedbackStore(
            "unused-local-dir", s3=s3, s3_prefix="feedback/"
        )
        app.add_middleware(RequestLoggingMiddleware)
        register_exception_handlers(app)
        app.include_router(router, prefix="/api/v1")
        app.add_api_route("/health", health, methods=["GET"])

        async def _override_get_db() -> Any:
            yield None

        app.dependency_overrides[get_db] = _override_get_db
        with TestClient(app) as client:
            yield client, s3


def test_predict_via_s3_key(s3_app) -> None:
    client, _ = s3_app
    response = client.post("/api/v1/predict", data={"s3_key": "predict-audio/clip1.wav"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_classes"]
    assert len(payload["all_scores"]) == 7


def test_predict_s3_missing_key_returns_400(s3_app) -> None:
    client, _ = s3_app
    response = client.post("/api/v1/predict", data={"s3_key": "predict-audio/nope.wav"})
    assert response.status_code == 400
    assert response.json()["error_code"] == "S3_DOWNLOAD_FAILED"


def test_predict_no_audio_source_returns_422(s3_app) -> None:
    client, _ = s3_app
    response = client.post("/api/v1/predict", data={})
    assert response.status_code == 422
    assert response.json()["error_code"] == "MISSING_AUDIO"


def test_feedback_stored_to_s3(s3_app, fixture_wav_path: Path) -> None:
    client, s3 = s3_app
    audio_b64 = base64.b64encode(fixture_wav_path.read_bytes()).decode("ascii")
    response = client.post(
        "/api/v1/feedback",
        json={
            "audio_base64": audio_b64,
            "correct_labels": ["blocks"],
            "original_prediction": ["fluent"],
            "model_version": "v3.0",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    keys = s3.list_keys("feedback/")
    assert any(k.endswith(".json") for k in keys)
    assert any(k.endswith(".wav") for k in keys)
