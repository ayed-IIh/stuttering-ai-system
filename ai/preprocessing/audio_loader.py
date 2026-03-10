"""Audio loading utilities for preprocessing speech recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


def load_wav(file_path: str | Path, target_sr: int = 16_000) -> Tuple[np.ndarray, int]:
    """Load an audio file and return waveform + sample rate."""
    waveform, sample_rate = librosa.load(str(file_path), sr=target_sr, mono=True)
    return waveform, sample_rate


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """Apply amplitude normalization for stable downstream features."""
    peak = np.max(np.abs(waveform))
    if peak == 0:
        return waveform
    return waveform / peak


if __name__ == "__main__":
    # Minimal local smoke check
    print("audio_loader ready")
