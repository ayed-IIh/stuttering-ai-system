from __future__ import annotations

import base64
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


def test_predict_returns_mobile_multilabel_contract(client, fixture_wav_path: Path) -> None:
    """The /predict response must carry the multi-label-shaped fields the mobile
    client requires (predicted_classes[], all_scores, threshold) — single-label
    output maps to exactly one predicted_classes entry."""
    data = fixture_wav_path.read_bytes()
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("silence.wav", data, "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()

    # predicted_classes: list of objects, each with `class` + `confidence`
    assert isinstance(payload["predicted_classes"], list)
    assert len(payload["predicted_classes"]) == 1
    entry = payload["predicted_classes"][0]
    assert entry["class"] == payload["predicted_class"]  # serialized via alias
    assert isinstance(entry["confidence"], float)

    # all_scores: full 0-1 map of every class
    assert isinstance(payload["all_scores"], dict)
    assert len(payload["all_scores"]) == NUM_CLASSES
    assert all(0.0 <= v <= 1.0 for v in payload["all_scores"].values())

    # threshold equals the winner's confidence (single-label consistency)
    winner_conf = payload["all_scores"][payload["predicted_class"]]
    assert payload["threshold"] == winner_conf
    assert entry["confidence"] == winner_conf


def test_feedback_accepts_correction(client, fixture_wav_path: Path) -> None:
    """The /feedback endpoint stores a therapist correction and returns accepted."""
    audio_b64 = base64.b64encode(fixture_wav_path.read_bytes()).decode("ascii")
    response = client.post(
        "/api/v1/feedback",
        json={
            "audio_base64": audio_b64,
            "correct_labels": ["blocks", "prolongations"],
            "original_prediction": ["fluent"],
            "model_version": "v3.0",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["feedback_id"]
    assert payload["stored_count"] >= 1


def test_feedback_rejects_invalid_base64(client) -> None:
    response = client.post(
        "/api/v1/feedback",
        json={
            "audio_base64": "!!!not-base64!!!",
            "correct_labels": ["blocks"],
            "original_prediction": ["fluent"],
            "model_version": "v3.0",
        },
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_FEEDBACK"


def test_feedback_rejects_non_wav_audio(client) -> None:
    """Valid base64 but non-WAV bytes must be rejected (corpus integrity)."""
    audio_b64 = base64.b64encode(b"not a wav file body at all").decode("ascii")
    response = client.post(
        "/api/v1/feedback",
        json={
            "audio_base64": audio_b64,
            "correct_labels": ["blocks"],
            "original_prediction": [],
            "model_version": "v3.0",
        },
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_FEEDBACK"


def test_feedback_rejects_unknown_label(client, fixture_wav_path: Path) -> None:
    """A correction with a label outside the taxonomy must be rejected."""
    audio_b64 = base64.b64encode(fixture_wav_path.read_bytes()).decode("ascii")
    response = client.post(
        "/api/v1/feedback",
        json={
            "audio_base64": audio_b64,
            "correct_labels": ["not_a_real_class"],
            "original_prediction": ["fluent"],
            "model_version": "v3.0",
        },
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_FEEDBACK"


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
