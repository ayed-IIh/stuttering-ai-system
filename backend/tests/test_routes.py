from __future__ import annotations

from pathlib import Path

from shared.labels import NUM_CLASSES


def test_health_returns_200(client) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert "version" in payload


def test_classes_returns_5_labels(client) -> None:
    """Canonical taxonomy length (repo uses seven labels; SDQ text referenced five)."""
    response = client.get("/api/v1/classes")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["classes"]) == NUM_CLASSES
    assert payload["label_to_id"]["fluent"] == 0
    assert payload["id_to_label"]["0"] == "fluent"


def test_predict_valid_wav_returns_prediction(client, fixture_wav_path: Path) -> None:
    data = fixture_wav_path.read_bytes()
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("silence.wav", data, "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "fluent"
    assert "confidence_scores" in payload
    assert len(payload["confidence_scores"]) == NUM_CLASSES
    assert payload["request_id"]
    assert payload["model_version"] == "mock-v1"


def test_predict_invalid_mime_type_returns_400(client, fixture_wav_path: Path) -> None:
    """Invalid MIME is rejected with 415 Unsupported Media Type (middleware contract)."""
    data = fixture_wav_path.read_bytes()
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("silence.wav", data, "audio/mpeg")},
    )
    assert response.status_code == 415
    body = response.json()
    assert body["error_code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_predict_missing_file_returns_422(client) -> None:
    response = client.post("/api/v1/predict", files={})
    assert response.status_code == 422


def test_predict_corrupt_audio_returns_422(client) -> None:
    corrupt = b"RIFF" + b"not-a-valid-wav-body"
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("corrupt.wav", corrupt, "audio/wav")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "UNPROCESSABLE_AUDIO"


def test_predict_oversized_file_returns_413(client) -> None:
    oversized = b"RIFF" + (b"\x00" * (2 * 1024 * 1024))
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("big.wav", oversized, "audio/wav")},
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "FILE_TOO_LARGE"
