"""
Audio augmentation transforms for stuttering-speech training data.

All transforms expect waveform shape (1, N), dtype torch.float32, and return
the same shape. Augmentations are training-only; never apply at inference.

Stuttering safety:
  add_gaussian_noise    — SAFE: additive noise leaves disfluency timing intact.
  time_shift            — SAFE: rigid translation preserves all temporal patterns.
  speed_perturbation    — USE WITH CAUTION: compresses/expands the temporal
                          envelope; block duration and repetition rate shift
                          proportionally. Acceptable within ±10% (rate_range
                          defaulting to [0.9, 1.1]).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio.functional as taf


def _validate_waveform(waveform: object) -> None:
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(
            f"waveform must be a torch.Tensor, got {type(waveform).__name__}"
        )
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(
            f"waveform must have shape (1, N), got {tuple(waveform.shape)}"
        )
    if waveform.dtype != torch.float32:
        raise TypeError(
            f"waveform must be float32, got {waveform.dtype}"
        )


def add_gaussian_noise(
    waveform: torch.Tensor,
    snr_db: float = 20.0,
) -> torch.Tensor:
    """Add Gaussian noise at a given signal-to-noise ratio.

    Args:
        waveform: Shape (1, N), float32, range roughly [-1, 1].
        snr_db:   Target SNR in decibels. Higher = cleaner signal.

    Returns:
        Noisy waveform, same shape and dtype. Not clamped — clamping
        would alter the effective SNR for loud clips.
    """
    _validate_waveform(waveform)

    signal_power = waveform.pow(2).mean()
    # SNR(dB) = 10*log10(P_signal / P_noise)  →  P_noise = P_signal / 10^(snr/10)
    noise_std = (signal_power / 10.0 ** (snr_db / 10.0)).sqrt()
    noise = torch.randn_like(waveform) * noise_std

    return waveform + noise


def time_shift(
    waveform: torch.Tensor,
    sr: int,
    shift_max_ms: float = 200.0,
) -> torch.Tensor:
    """Shift audio in time by a random offset, zero-filling the gap.

    Args:
        waveform:      Shape (1, N), float32.
        sr:            Sample rate in Hz.
        shift_max_ms:  Maximum absolute shift in milliseconds. The actual
                       shift is drawn uniformly from [-shift_max_ms, +shift_max_ms].

    Returns:
        Shifted waveform, same shape. Positive shift → silence prepended;
        negative shift → silence appended.
    """
    _validate_waveform(waveform)

    max_samples = int(sr * shift_max_ms / 1000.0)
    if max_samples == 0:
        return waveform.clone()

    shift = torch.randint(-max_samples, max_samples + 1, (1,)).item()
    out = torch.roll(waveform, shift, dims=-1)

    # torch.roll wraps values around the boundary; zero those out.
    if shift > 0:
        out[..., :shift] = 0.0
    elif shift < 0:
        out[..., shift:] = 0.0

    return out


def speed_perturbation(
    waveform: torch.Tensor,
    sr: int,
    rate_range: tuple[float, float] = (0.9, 1.1),
) -> torch.Tensor:
    """Change playback speed without altering pitch.

    Uses the torchaudio resampling trick: interpreting N samples as if they
    were recorded at sr*rate Hz, then resampling back to sr, effectively
    speeds up or slows down playback without the pitch shift that naive
    resampling produces.

    Args:
        waveform:    Shape (1, N), float32.
        sr:          Sample rate in Hz.
        rate_range:  (min_rate, max_rate). Rate > 1.0 speeds up; < 1.0 slows.

    Returns:
        Perturbed waveform, exactly the same shape as input.
    """
    _validate_waveform(waveform)

    rate = torch.empty(1).uniform_(*rate_range).item()
    if rate == 1.0:
        return waveform.clone()

    orig_len = waveform.shape[-1]
    # Treat the existing samples as if recorded at a higher/lower rate,
    # then pull them back to sr — this warps time without touching pitch.
    fake_sr = int(sr * rate)
    perturbed = taf.resample(waveform, orig_freq=fake_sr, new_freq=sr)

    curr_len = perturbed.shape[-1]
    if curr_len < orig_len:
        perturbed = F.pad(perturbed, (0, orig_len - curr_len))
    elif curr_len > orig_len:
        start = (curr_len - orig_len) // 2
        perturbed = perturbed[..., start: start + orig_len]

    return perturbed


# Iteration order is fixed — reproducible under a manual seed.
_AUGMENTATION_FNS: dict = {
    "gaussian_noise": lambda w, _sr, p: add_gaussian_noise(w, p["snr_db"]),
    "time_shift": lambda w, sr, p: time_shift(w, sr, p["shift_max_ms"]),
    "speed_perturbation": lambda w, sr, p: speed_perturbation(
        w, sr, tuple(p["rate_range"])  # coerce list (JSON) to tuple
    ),
}


def apply_augmentations(
    waveform: torch.Tensor,
    sr: int,
    config: dict,
) -> torch.Tensor:
    """Apply a configured set of augmentations, each with its own probability.

    Each augmentation fires independently. The order is gaussian_noise →
    time_shift → speed_perturbation, regardless of key order in config.

    Args:
        waveform: Shape (1, N), float32.
        sr:       Sample rate in Hz.
        config:   Mapping of augmentation name → parameter dict. Each
                  parameter dict must include "prob" (float in [0, 1]).
                  Unknown keys are silently ignored. Example::

                      {
                          "gaussian_noise":     {"prob": 0.3, "snr_db": 20.0},
                          "time_shift":         {"prob": 0.5, "shift_max_ms": 200},
                          "speed_perturbation": {"prob": 0.3, "rate_range": [0.9, 1.1]},
                      }

    Returns:
        Augmented waveform, same shape and dtype as input.
    """
    _validate_waveform(waveform)

    for name, fn in _AUGMENTATION_FNS.items():
        params = config.get(name)
        if params is None:
            continue
        if torch.rand(1).item() < params["prob"]:
            waveform = fn(waveform, sr, params)

    return waveform
