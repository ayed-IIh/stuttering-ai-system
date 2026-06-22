from __future__ import annotations

import numpy as np

from backend.services.audio_preprocess import analyze_quality, clean

SR = 16000


def _tone(seconds: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_quality_ok_for_a_normal_clip() -> None:
    q = analyze_quality(_tone(1.0), SR)
    assert q["ok"] is True
    assert q["warnings"] == []
    assert abs(q["duration_sec"] - 1.0) < 0.01


def test_quality_flags_silence() -> None:
    q = analyze_quality(np.zeros(SR, dtype=np.float32), SR)
    assert q["ok"] is False
    assert "mostly_silence" in q["warnings"]
    assert "audio_too_quiet" in q["warnings"]


def test_quality_flags_too_short() -> None:
    q = analyze_quality(_tone(0.1), SR)
    assert "audio_too_short" in q["warnings"]


def test_quality_flags_clipping() -> None:
    x = np.ones(SR, dtype=np.float32)  # fully clipped
    q = analyze_quality(x, SR)
    assert "audio_clipped" in q["warnings"]


def test_clean_normalizes_peak_and_keeps_speech() -> None:
    quiet = _tone(1.0, amp=0.05)
    out = clean(quiet, SR)
    assert out.dtype == np.float32
    assert out.size > 0
    assert abs(float(np.max(np.abs(out))) - 0.95) < 0.02  # peak-normalized


def test_clean_empty_is_safe() -> None:
    out = clean(np.array([], dtype=np.float32), SR)
    assert out.size == 0
