"""Tests for backend.services.model_service (inference + audio preprocessing).

Note: many of these tests use ``torchaudio.save`` to produce WAV fixtures.
On torchaudio>=2.8 that path requires the ``torchcodec`` native runtime
(FFmpeg). Skip the whole module when it isn't usable so the rest of the
suite stays green on machines without FFmpeg.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import torch
import torchaudio

try:  # noqa: SIM105
    import torchcodec  # noqa: F401
except (ImportError, OSError, RuntimeError) as _exc:  # pragma: no cover - env-dependent
    pytest.skip(
        f"torchcodec runtime unavailable ({_exc.__class__.__name__}); "
        f"audio fixtures cannot be written. Install FFmpeg and torchcodec "
        f"to enable.",
        allow_module_level=True,
    )

from ai.preprocessing.audio_loader import TARGET_SAMPLES
from backend.app.config import Settings, clear_settings_cache
from backend.services.model_service import (
    InvalidAudioError,
    ModelNotLoadedError,
    ModelService,
)


def _write_wav(path: Path, *, sr: int, channels: int, seconds: float) -> None:
    t = int(sr * seconds)
    if channels == 1:
        x = torch.sin(2 * 3.14159 * 440.0 * torch.linspace(0, seconds, t)).unsqueeze(0)
    else:
        x = torch.stack(
            [
                torch.sin(2 * 3.14159 * 440.0 * torch.linspace(0, seconds, t)),
                torch.sin(2 * 3.14159 * 220.0 * torch.linspace(0, seconds, t)),
            ],
            dim=0,
        )
    torchaudio.save(str(path), x, sr)


@pytest.fixture
def three_wav_fixtures(tmp_path: Path) -> list[Path]:
    """Three distinct WAVs: 16 kHz mono, 8 kHz mono, 44.1 kHz stereo."""
    paths = [
        tmp_path / "s16_mono.wav",
        tmp_path / "s8_mono.wav",
        tmp_path / "s441_stereo.wav",
    ]
    _write_wav(paths[0], sr=16_000, channels=1, seconds=0.4)
    _write_wav(paths[1], sr=8_000, channels=1, seconds=0.35)
    _write_wav(paths[2], sr=44_100, channels=2, seconds=0.25)
    return paths


@pytest.fixture
def fallback_settings() -> Settings:
    clear_settings_cache()
    s = Settings(
        MODEL_PATH="",
        MODEL_SOURCE="local",
        DEVICE="cpu",
        MAX_AUDIO_SIZE_MB=10,
        allowed_origins_raw="",
        PRODUCTION_MODE=False,
        DB_URL="postgresql://localhost/x",
        SERVICE_VERSION="test-0",
        LOG_LEVEL="INFO",
    )
    yield s
    clear_settings_cache()


def test_predict_fallback_three_wav_files(fallback_settings: Settings, three_wav_fixtures: list[Path]):
    svc = ModelService(fallback_settings)
    assert svc.is_loaded()
    for p in three_wav_fixtures:
        data = p.read_bytes()
        out = svc.predict(data)
        assert set(out.keys()) == {
            "predicted_class",
            "confidence_scores",
            "processing_time_ms",
            "model_version",
        }
        assert isinstance(out["confidence_scores"], dict)
        assert out["model_version"] == "test-0"
        assert out["processing_time_ms"] >= 0


def test_preprocess_matches_audio_loader_target_length(fallback_settings: Settings, three_wav_fixtures: list[Path]):
    svc = ModelService(fallback_settings)
    raw = three_wav_fixtures[0].read_bytes()
    wf = svc._decode_and_preprocess_wav(raw)
    assert wf.shape == (1, TARGET_SAMPLES)
    assert wf.dtype == torch.float32


def test_invalid_audio_empty(fallback_settings: Settings):
    svc = ModelService(fallback_settings)
    with pytest.raises(InvalidAudioError):
        svc.predict(b"")


def test_invalid_audio_garbage(fallback_settings: Settings):
    svc = ModelService(fallback_settings)
    with pytest.raises(InvalidAudioError):
        svc.predict(b"not a wav file")


def test_predict_not_loaded_after_shutdown(fallback_settings: Settings, three_wav_fixtures: list[Path]):
    svc = ModelService(fallback_settings)
    svc.shutdown()
    assert not svc.is_loaded()
    with pytest.raises(ModelNotLoadedError):
        svc.predict(three_wav_fixtures[0].read_bytes())


def test_predict_thread_safe(fallback_settings: Settings, three_wav_fixtures: list[Path]):
    svc = ModelService(fallback_settings)
    blobs = [p.read_bytes() for p in three_wav_fixtures]
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for _ in range(5):
                svc.predict(blobs[i % len(blobs)])
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
