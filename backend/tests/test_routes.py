"""HTTP-level tests for the multi-label /predict, /classes, /health routes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from shared.labels import CLASS_LABELS, NUM_CLASSES


def test_health_returns_200(client) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert "version" in payload


def test_classes_returns_7_labels(client) -> None:
    """Canonical taxonomy length must equal NUM_CLASSES (7)."""
    response = client.get("/api/v1/classes")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["classes"]) == NUM_CLASSES
    assert payload["label_to_id"]["fluent"] == 0
    assert payload["id_to_label"]["0"] == "fluent"


def test_predict_valid_wav_returns_multi_label_response(
    client, fixture_wav_path: Path
) -> None:
    data = fixture_wav_path.read_bytes()
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("silence.wav", data, "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    # New multi-label envelope
    assert "predicted_classes" in payload
    assert "all_scores" in payload
    assert "threshold" in payload
    assert "model_version" in payload
    assert "request_id" in payload
    assert "processing_time_ms" in payload
    # Old single-label fields must be absent
    assert "predicted_class" not in payload
    assert "confidence_scores" not in payload
    # all_scores has exactly the 7 keys
    assert set(payload["all_scores"].keys()) == set(CLASS_LABELS)
    assert payload["model_version"] == "mock-v1"


def test_predict_zero_classes_above_threshold_returns_empty_list(
    client, fixture_wav_path: Path
) -> None:
    """Default mock has every score at 1/7 (~0.143) < threshold 0.5."""
    data = fixture_wav_path.read_bytes()
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("silence.wav", data, "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["predicted_classes"] == []


def test_predict_two_classes_above_threshold(
    client, test_app, fixture_wav_path: Path
) -> None:
    """When the mock returns two classes above threshold, both appear."""
    mock = test_app.state.model_service
    mock.next_scores = {label: 0.0 for label in CLASS_LABELS}
    mock.next_scores["blocks"] = 0.9
    mock.next_scores["prolongations"] = 0.7
    data = fixture_wav_path.read_bytes()
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("silence.wav", data, "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    names = [c["class"] for c in payload["predicted_classes"]]
    assert "blocks" in names
    assert "prolongations" in names
    # Sorted by descending confidence
    assert payload["predicted_classes"][0]["class"] == "blocks"
    assert payload["predicted_classes"][1]["class"] == "prolongations"
    mock.next_scores = None  # reset


def test_predict_all_seven_classes_above_threshold(
    client, test_app, fixture_wav_path: Path
) -> None:
    mock = test_app.state.model_service
    mock.next_scores = {label: 0.99 for label in CLASS_LABELS}
    data = fixture_wav_path.read_bytes()
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("silence.wav", data, "audio/wav")},
    )
    assert response.status_code == 200
    assert len(response.json()["predicted_classes"]) == NUM_CLASSES
    mock.next_scores = None


def test_predict_handles_two_simultaneous_requests(client, fixture_wav_path: Path) -> None:
    data = fixture_wav_path.read_bytes()

    def post_predict():
        return client.post(
            "/api/v1/predict",
            files={"audio_file": ("silence.wav", data, "audio/wav")},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: post_predict(), range(2)))

    assert [response.status_code for response in responses] == [200, 200]
    for response in responses:
        assert set(response.json()["all_scores"].keys()) == set(CLASS_LABELS)


def test_predict_invalid_mime_type_returns_415(
    client, fixture_wav_path: Path
) -> None:
    """Invalid MIME is rejected with 415 Unsupported Media Type."""
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


def test_all_scores_has_exactly_seven_keys(
    client, fixture_wav_path: Path
) -> None:
    data = fixture_wav_path.read_bytes()
    response = client.post(
        "/api/v1/predict",
        files={"audio_file": ("silence.wav", data, "audio/wav")},
    )
    assert response.status_code == 200
    assert len(response.json()["all_scores"]) == NUM_CLASSES
