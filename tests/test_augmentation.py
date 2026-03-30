"""
Tests for ai/preprocessing/augmentation.py.

Synthetic waveforms only — no real .wav files needed.
All RNG is controlled via torch.manual_seed so tests are deterministic.
"""

import pytest
import torch

from ai.preprocessing.augmentation import (
    add_gaussian_noise,
    apply_augmentations,
    speed_perturbation,
    time_shift,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wav():
    """1 second of random audio at 16 kHz, channel-first float32."""
    torch.manual_seed(0)
    return torch.randn(1, 16_000, dtype=torch.float32)


@pytest.fixture
def sr():
    return 16_000


@pytest.fixture
def config():
    """Full augmentation config — used to test apply_augmentations."""
    return {
        "gaussian_noise":     {"prob": 0.5, "snr_db": 20.0},
        "time_shift":         {"prob": 0.5, "shift_max_ms": 200},
        "speed_perturbation": {"prob": 0.5, "rate_range": [0.9, 1.1]},
    }


# ---------------------------------------------------------------------------
# add_gaussian_noise
# ---------------------------------------------------------------------------

def test_gaussian_noise_shape(wav):
    """Output shape matches input shape."""
    out = add_gaussian_noise(wav)
    assert out.shape == wav.shape


def test_gaussian_noise_dtype(wav):
    """Output dtype is float32."""
    assert add_gaussian_noise(wav).dtype == torch.float32


def test_gaussian_noise_changes_signal(wav):
    """Noise is actually added — output differs from input."""
    torch.manual_seed(1)
    out = add_gaussian_noise(wav, snr_db=20.0)
    assert not torch.equal(out, wav)


def test_gaussian_noise_deterministic(wav):
    """Same seed produces identical output."""
    torch.manual_seed(42)
    out1 = add_gaussian_noise(wav, snr_db=20.0)
    torch.manual_seed(42)
    out2 = add_gaussian_noise(wav, snr_db=20.0)
    assert torch.allclose(out1, out2)


def test_gaussian_noise_silent_input(sr):
    """Silent clip stays silent — signal power is zero so noise std is zero."""
    silent = torch.zeros(1, sr, dtype=torch.float32)
    out = add_gaussian_noise(silent, snr_db=20.0)
    assert torch.allclose(out, silent)


def test_gaussian_noise_high_snr_small_perturbation(wav):
    """At 60 dB SNR the perturbation is imperceptibly small."""
    out = add_gaussian_noise(wav, snr_db=60.0)
    assert (out - wav).abs().mean().item() < 1e-3


# ---------------------------------------------------------------------------
# time_shift
# ---------------------------------------------------------------------------

def test_time_shift_shape(wav, sr):
    """Output shape matches input shape."""
    torch.manual_seed(1)
    out = time_shift(wav, sr)
    assert out.shape == wav.shape


def test_time_shift_dtype(wav, sr):
    """Output dtype is float32."""
    torch.manual_seed(1)
    assert time_shift(wav, sr).dtype == torch.float32


def test_time_shift_zero_ms_returns_clone(wav, sr):
    """Zero shift_max_ms returns a copy of the input without touching RNG."""
    out = time_shift(wav, sr, shift_max_ms=0)
    assert torch.equal(out, wav)
    assert out is not wav


def test_time_shift_zero_fill_leading(wav, sr):
    """Positive shift zeroes the leading samples."""
    # Force a known positive shift by seeding and reading the same draw.
    torch.manual_seed(7)
    out = time_shift(wav, sr, shift_max_ms=100)

    torch.manual_seed(7)
    max_samples = int(sr * 100 / 1000)
    shift = torch.randint(-max_samples, max_samples + 1, (1,)).item()

    if shift > 0:
        assert out[..., :shift].abs().max().item() == 0.0
    elif shift < 0:
        assert out[..., shift:].abs().max().item() == 0.0
    # shift == 0: nothing to assert about zero-fill, output equals input


def test_time_shift_deterministic(wav, sr):
    """Same seed produces identical output."""
    torch.manual_seed(3)
    out1 = time_shift(wav, sr, shift_max_ms=200)
    torch.manual_seed(3)
    out2 = time_shift(wav, sr, shift_max_ms=200)
    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# speed_perturbation
# ---------------------------------------------------------------------------

def test_speed_perturbation_shape(wav, sr):
    """Output shape is exactly (1, 16000) — critical contract."""
    torch.manual_seed(1)
    out = speed_perturbation(wav, sr)
    assert out.shape == wav.shape


def test_speed_perturbation_dtype(wav, sr):
    """Output dtype is float32."""
    torch.manual_seed(1)
    assert speed_perturbation(wav, sr).dtype == torch.float32


def test_speed_perturbation_rate_one_returns_clone(wav, sr):
    """rate=1.0 returns a copy of the input unchanged."""
    out = speed_perturbation(wav, sr, rate_range=(1.0, 1.0))
    assert torch.equal(out, wav)
    assert out is not wav


def test_speed_perturbation_faster_preserves_shape(wav, sr):
    """rate > 1.0 (speed up) still returns the same shape."""
    out = speed_perturbation(wav, sr, rate_range=(1.1, 1.1))
    assert out.shape == wav.shape


def test_speed_perturbation_slower_preserves_shape(wav, sr):
    """rate < 1.0 (slow down) still returns the same shape."""
    out = speed_perturbation(wav, sr, rate_range=(0.9, 0.9))
    assert out.shape == wav.shape


def test_speed_perturbation_no_nan_or_inf(wav, sr):
    """No NaN or Inf in the output for any valid rate."""
    torch.manual_seed(5)
    out = speed_perturbation(wav, sr)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# apply_augmentations
# ---------------------------------------------------------------------------

def test_apply_prob_zero_returns_input(wav, sr):
    """prob=0 for every augmentation → output equals input."""
    cfg = {
        "gaussian_noise":     {"prob": 0.0, "snr_db": 20.0},
        "time_shift":         {"prob": 0.0, "shift_max_ms": 200},
        "speed_perturbation": {"prob": 0.0, "rate_range": [0.9, 1.1]},
    }
    out = apply_augmentations(wav, sr, cfg)
    assert torch.allclose(out, wav)


def test_apply_prob_one_shape(wav, sr):
    """prob=1 for all → output shape is unchanged."""
    cfg = {
        "gaussian_noise":     {"prob": 1.0, "snr_db": 20.0},
        "time_shift":         {"prob": 1.0, "shift_max_ms": 200},
        "speed_perturbation": {"prob": 1.0, "rate_range": [0.9, 1.1]},
    }
    torch.manual_seed(1)
    out = apply_augmentations(wav, sr, cfg)
    assert out.shape == wav.shape


def test_apply_prob_one_dtype(wav, sr):
    """prob=1 for all → output dtype is float32."""
    cfg = {
        "gaussian_noise":     {"prob": 1.0, "snr_db": 20.0},
        "time_shift":         {"prob": 1.0, "shift_max_ms": 200},
        "speed_perturbation": {"prob": 1.0, "rate_range": [0.9, 1.1]},
    }
    torch.manual_seed(1)
    out = apply_augmentations(wav, sr, cfg)
    assert out.dtype == torch.float32


def test_apply_prob_one_changes_signal(wav, sr):
    """prob=1 for all → at least one transform actually ran."""
    cfg = {
        "gaussian_noise":     {"prob": 1.0, "snr_db": 20.0},
        "time_shift":         {"prob": 1.0, "shift_max_ms": 200},
        "speed_perturbation": {"prob": 1.0, "rate_range": [0.9, 1.1]},
    }
    torch.manual_seed(2)
    out = apply_augmentations(wav, sr, cfg)
    assert not torch.equal(out, wav)


def test_apply_empty_config_returns_input(wav, sr):
    """Empty config is a no-op."""
    out = apply_augmentations(wav, sr, {})
    assert torch.allclose(out, wav)


def test_apply_deterministic(wav, sr, config):
    """Same seed produces identical output."""
    torch.manual_seed(99)
    out1 = apply_augmentations(wav, sr, config)
    torch.manual_seed(99)
    out2 = apply_augmentations(wav, sr, config)
    assert torch.allclose(out1, out2)


def test_apply_list_rate_range(wav, sr):
    """rate_range as a list (e.g. from JSON) is accepted without error."""
    cfg = {"speed_perturbation": {"prob": 1.0, "rate_range": [0.95, 1.05]}}
    torch.manual_seed(1)
    out = apply_augmentations(wav, sr, cfg)
    assert out.shape == wav.shape


# ---------------------------------------------------------------------------
# _validate_waveform — exercised via public API
# ---------------------------------------------------------------------------

def test_validate_non_tensor_raises():
    """Non-tensor input raises TypeError."""
    with pytest.raises(TypeError):
        add_gaussian_noise([0.1, 0.2, 0.3])


def test_validate_1d_raises(sr):
    """1-D tensor (N,) raises ValueError."""
    with pytest.raises(ValueError):
        time_shift(torch.randn(16_000), sr)


def test_validate_multichannel_raises(sr):
    """Shape (2, N) raises ValueError."""
    with pytest.raises(ValueError):
        speed_perturbation(torch.randn(2, 16_000), sr)


def test_validate_wrong_dtype_raises(sr):
    """float64 input raises TypeError."""
    with pytest.raises(TypeError):
        add_gaussian_noise(torch.randn(1, 16_000, dtype=torch.float64))
