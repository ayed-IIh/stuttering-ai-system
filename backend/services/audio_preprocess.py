"""Audio pre-processing for the inference path.

Two independent stages:

1. ``analyze_quality`` — a SAFE quality gate. It only MEASURES the (resampled,
   mono) waveform and returns metrics + warnings (too short / too quiet /
   clipped / mostly silence). It never changes the signal the model sees, so it
   carries zero train/serve-skew risk and is always on. The warnings let the
   mobile/therapist treat a poor recording cautiously.

2. ``clean`` — optional signal transforms (high-pass to drop rumble, silence
   trim, peak-normalize). These CHANGE the waveform, so enabling them at
   inference without matching the training pipeline can shift accuracy. Gated
   behind ``PREPROCESS_ENABLED`` (default off) and validated on the test set
   before use.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Quality thresholds (tunable via the call site if needed).
MIN_DURATION_SEC = 0.30
QUIET_RMS_DBFS = -45.0
CLIP_RATIO_WARN = 0.02
SILENCE_RATIO_WARN = 0.90


def _as_mono_1d(wav: Any) -> np.ndarray:
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def analyze_quality(wav: Any, sr: int) -> dict:
    """Measure recording quality WITHOUT altering the signal."""
    x = _as_mono_1d(wav)
    n = int(x.size)
    duration = n / sr if sr else 0.0
    peak = float(np.max(np.abs(x))) if n else 0.0
    rms = float(np.sqrt(np.mean(np.square(x)))) if n else 0.0
    rms_dbfs = float(20.0 * np.log10(rms + 1e-9))
    clip_ratio = float(np.mean(np.abs(x) >= 0.99)) if n else 0.0

    # Silence ratio over 20 ms frames.
    frame = max(1, int(0.02 * sr)) if sr else 1
    if n >= frame:
        usable = x[: n - (n % frame)].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(np.square(usable), axis=1))
        silence_ratio = float(np.mean(frame_rms < 0.01))
    else:
        silence_ratio = 1.0

    warnings: list[str] = []
    if duration < MIN_DURATION_SEC:
        warnings.append("audio_too_short")
    if rms_dbfs < QUIET_RMS_DBFS:
        warnings.append("audio_too_quiet")
    if clip_ratio > CLIP_RATIO_WARN:
        warnings.append("audio_clipped")
    if silence_ratio > SILENCE_RATIO_WARN:
        warnings.append("mostly_silence")

    return {
        "duration_sec": round(duration, 3),
        "rms_dbfs": round(rms_dbfs, 1),
        "peak": round(peak, 3),
        "clipping_ratio": round(clip_ratio, 4),
        "silence_ratio": round(silence_ratio, 3),
        "warnings": warnings,
        "ok": len(warnings) == 0,
    }


def clean(
    wav: Any,
    sr: int,
    *,
    high_pass: bool = True,
    trim: bool = True,
    normalize: bool = True,
    hp_cutoff_hz: float = 80.0,
    trim_top_db: float = 30.0,
    target_peak: float = 0.95,
) -> np.ndarray:
    """Optional signal cleanup: high-pass → silence-trim → peak-normalize.

    Returns the processed mono float32 waveform. Each stage is best-effort: if a
    stage can't run, the signal passes through unchanged rather than failing.
    """
    x = _as_mono_1d(wav)
    if x.size == 0:
        return x

    if high_pass and sr and sr > 2 * hp_cutoff_hz:
        from scipy.signal import butter, sosfilt

        sos = butter(4, hp_cutoff_hz / (sr / 2.0), btype="highpass", output="sos")
        x = sosfilt(sos, x).astype(np.float32)

    if trim:
        import librosa

        trimmed, _ = librosa.effects.trim(x, top_db=trim_top_db)
        if trimmed.size:
            x = trimmed.astype(np.float32)

    if normalize:
        peak = float(np.max(np.abs(x)))
        if peak > 1e-6:
            x = (x * (target_peak / peak)).astype(np.float32)

    return x
