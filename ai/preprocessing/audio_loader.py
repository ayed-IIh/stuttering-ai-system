"""
audio_loader.py — Foundational audio loading and preprocessing utilities.

Pipeline (applied in this order by load_audio):
  1. Resample to target_sr (default 16 kHz — Wav2Vec2/HuBERT pretraining assumption)
  2. Stereo → mono via channel mean (preserves energy)
  3. Peak normalize to [-1.0, 1.0]
  4. Trim silence — training only (inference keeps leading silence: clinically meaningful)
  5. Fixed duration: center-crop long clips, right-pad short clips
  6. Cast to float32 (PyTorch default compute dtype)

Output contract (guaranteed by load_audio):
  shape=(1, 160000), dtype=torch.float32, range=[-1.0, 1.0], sr=16000
"""

import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

logger = logging.getLogger(__name__)

# Fixed output length: 10 seconds × 16 000 Hz
TARGET_DURATION_SEC: float = 10.0
TARGET_SAMPLES: int = 160_000


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_tensor(obj: object, name: str = "waveform") -> None:
    """Raise TypeError if obj is not a torch.Tensor."""
    if not isinstance(obj, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(obj).__name__}"
        )


def _validate_sr(sr: object) -> None:
    """Raise ValueError if sr is not a positive int."""
    if not isinstance(sr, int) or sr <= 0:
        raise ValueError(f"sr must be a positive int, got {sr!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_audio(
    file_path: str,
    target_sr: int = 16_000,
    training: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Load a .wav file and run the full preprocessing pipeline.

    Steps applied in order:
      1. Decode with torchaudio (float32 waveform)
      2. Stereo → mono via channel mean
      3. Resample to target_sr if the native rate differs
      4. Peak-normalize to [-1.0, 1.0]
      5. Trim leading/trailing silence (training=True only)
      6. Center-crop or right-pad to TARGET_SAMPLES
      7. Ensure float32 dtype

    Args:
        file_path:  Absolute or relative path to a .wav file.
        target_sr:  Desired sample rate (default 16 000 Hz).
        training:   When True, silence trimming (step 5) is enabled.

    Returns:
        waveform: shape (1, TARGET_SAMPLES), dtype float32, range [-1, 1].
        sample_rate: target_sr (always).

    Raises:
        TypeError:        file_path is not a str.
        FileNotFoundError: path does not exist.
        ValueError:       file is not a .wav.
    """
    if not isinstance(file_path, str):
        raise TypeError(
            f"file_path must be a str, got {type(file_path).__name__}"
        )

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file, got suffix '{path.suffix}'")

    # Step 1 — decode
    waveform, native_sr = torchaudio.load(file_path)  # (channels, time)
    logger.debug("Loaded %s: shape=%s sr=%d", path.name, waveform.shape, native_sr)

    # Step 2 — stereo → mono (mean over channel dim preserves total energy)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Step 3 — resample if needed
    if native_sr != target_sr:
        waveform = F.resample(waveform, orig_freq=native_sr, new_freq=target_sr)
        logger.debug("Resampled %d → %d Hz", native_sr, target_sr)

    # Step 4 — peak normalize
    waveform = normalize_waveform(waveform, method="peak")

    # Step 5 — silence trimming (training only)
    if training:
        waveform = trim_silence(waveform, sr=target_sr)

    # Step 6 — fixed duration
    waveform = pad_or_truncate(waveform, sr=target_sr, max_duration_sec=TARGET_DURATION_SEC)

    # Step 7 — enforce float32
    waveform = waveform.to(torch.float32)

    return waveform, target_sr


def normalize_waveform(
    waveform: torch.Tensor,
    method: str = "peak",
) -> torch.Tensor:
    """Normalize waveform amplitude.

    Args:
        waveform: Any shape torch.Tensor.
        method:   'peak' — divide by abs max + epsilon (matches LibriSpeech
                           pretraining distribution for Wav2Vec2/HuBERT).
                  'rms'  — divide by root-mean-square + epsilon.

    Returns:
        Normalized tensor with the same shape as input.

    Raises:
        TypeError:  waveform is not a torch.Tensor.
        ValueError: method is not 'peak' or 'rms'.
    """
    _validate_tensor(waveform)

    if method not in ("peak", "rms"):
        raise ValueError(
            f"method must be 'peak' or 'rms', got '{method}'"
        )

    if method == "peak":
        # Epsilon in the denominator — always applied, never conditional —
        # so silent clips pass through as all-zeros rather than blowing up.
        peak = waveform.abs().max()
        return waveform / (peak + 1e-8)

    # RMS: sqrt(mean(x²)) captures average loudness rather than peak loudness
    rms = waveform.pow(2).mean().sqrt()
    return waveform / (rms + 1e-8)


def trim_silence(
    waveform: torch.Tensor,
    sr: int,
    top_db: float = 30.0,
) -> torch.Tensor:
    """Trim leading and trailing silence using librosa.

    Note: called by load_audio only when training=True. Clinical recordings
    may carry meaningful pre-utterance silence that must be preserved at
    inference time.

    Args:
        waveform: Shape (1, time) or (time,) float32 tensor.
        sr:       Sample rate in Hz (unused by librosa.effects.trim but
                  kept for a consistent interface across utility functions).
        top_db:   Threshold (dB below peak) below which frames are silent.

    Returns:
        Trimmed tensor. Same number of dims as input, shorter in time axis.

    Raises:
        TypeError:  waveform is not a torch.Tensor.
        ValueError: sr is not a positive int.
    """
    _validate_tensor(waveform)
    _validate_sr(sr)

    # librosa expects a 1-D float numpy array
    audio_np: np.ndarray = waveform.squeeze().numpy()
    trimmed_np, _ = librosa.effects.trim(audio_np, top_db=top_db)

    trimmed = torch.from_numpy(trimmed_np)

    # Restore channel dim if input was (1, time)
    if waveform.dim() == 2:
        trimmed = trimmed.unsqueeze(0)

    return trimmed


def pad_or_truncate(
    waveform: torch.Tensor,
    sr: int,
    max_duration_sec: float = TARGET_DURATION_SEC,
) -> torch.Tensor:
    """Enforce a fixed audio duration.

    Long clips: center-crop. The first and last seconds often contain silence
    or mic noise, so discarding symmetrically keeps the core utterance.

    Short clips: right-pad with zeros. This matches the HuggingFace attention
    mask convention — padding is always appended at the end.

    Args:
        waveform:        Shape (..., time) tensor.
        sr:              Sample rate in Hz.
        max_duration_sec: Target duration in seconds (default 10.0 s).

    Returns:
        Tensor with shape[..., -1] == int(sr * max_duration_sec).

    Raises:
        TypeError:  waveform is not a torch.Tensor.
        ValueError: sr or max_duration_sec is non-positive.
    """
    _validate_tensor(waveform)
    _validate_sr(sr)

    if max_duration_sec <= 0:
        raise ValueError(
            f"max_duration_sec must be positive, got {max_duration_sec}"
        )

    target_length = int(sr * max_duration_sec)
    current_length = waveform.shape[-1]

    if current_length > target_length:
        # Center crop: discard equal amounts from both ends
        start = (current_length - target_length) // 2
        return waveform[..., start: start + target_length]

    if current_length < target_length:
        # Right-pad with zeros to reach target length
        pad_len = target_length - current_length
        padding = torch.zeros(
            *waveform.shape[:-1], pad_len, dtype=waveform.dtype
        )
        return torch.cat([waveform, padding], dim=-1)

    # Already exact length — no-op
    return waveform
