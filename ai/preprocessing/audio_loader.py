"""Audio loader and preprocessing helpers for stuttering classification."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


def load_wav(file_path: str | Path, target_sr: int = 16_000) -> Tuple[np.ndarray, int]:
    """Load `.wav` audio as mono waveform with a fixed sample rate."""
    waveform, sample_rate = librosa.load(str(file_path), sr=target_sr, mono=True)
    return waveform, sample_rate


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Normalize waveform amplitude to improve training stability."""
    peak = np.max(np.abs(waveform))
    if peak == 0:
        return waveform
    return waveform / peak


def to_float32(waveform: np.ndarray) -> np.ndarray:
    """Ensure model input dtype is float32 before tensor conversion."""
    return waveform.astype(np.float32, copy=False)


if __name__ == "__main__":
    # Placeholder smoke check; replace with real file-path tests.
    print("audio_loader module ready")