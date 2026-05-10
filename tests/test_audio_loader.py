"""
test_audio_loader.py — Unit tests for ai/preprocessing/audio_loader.py

All tests create real .wav files via torchaudio.save (no mocks). Starting
with **torchaudio 2.9** the WAV writer routes through ``torchcodec``, which
itself needs an FFmpeg native library at runtime. On older torchaudio
(<2.9) the soundfile/SoX backends handle WAV without FFmpeg, so we only
skip when both (a) torchaudio is 2.9+ AND (b) torchcodec isn't importable.
Run from repo root: pytest tests/test_audio_loader.py -v
"""

import pytest
import torch
import torchaudio

from packaging.version import InvalidVersion, Version

_TORCHCODEC_REQUIRED_FROM = Version("2.9.0")


def _torchaudio_needs_torchcodec() -> bool:
    """True iff installed torchaudio routes WAV save through torchcodec.

    Catches only ``InvalidVersion`` (raised by ``Version`` on PEP 440-invalid
    strings, e.g. dev/local-build tags like ``"2.9.0+cu121.dev"``). A
    catch-all here would mask import-time bugs.
    """
    try:
        return Version(torchaudio.__version__) >= _TORCHCODEC_REQUIRED_FROM
    except InvalidVersion:  # pragma: no cover - dev-version strings
        return True


if _torchaudio_needs_torchcodec():
    try:  # noqa: SIM105
        import torchcodec  # noqa: F401
    except (ImportError, OSError, RuntimeError) as _exc:  # pragma: no cover - env-dependent
        pytest.skip(
            f"torchaudio {torchaudio.__version__} requires torchcodec for "
            f"WAV save, and it is unavailable ({_exc.__class__.__name__}). "
            f"Install FFmpeg and torchcodec to enable.",
            allow_module_level=True,
        )

from ai.preprocessing.audio_loader import (
    TARGET_SAMPLES,
    load_audio,
    normalize_waveform,
    pad_or_truncate,
    trim_silence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path, num_frames: int, sr: int = 16_000, channels: int = 1) -> str:
    """Write a sine-wave .wav file and return its string path."""
    t = torch.linspace(0, num_frames / sr, num_frames)
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0).expand(channels, -1)
    torchaudio.save(str(path), waveform.float(), sr)
    return str(path)


# ---------------------------------------------------------------------------
# load_audio — happy path
# ---------------------------------------------------------------------------

def test_load_audio_returns_correct_shape(tmp_path):
    """load_audio must always return shape (1, 160000)."""
    wav = _make_wav(tmp_path / "audio.wav", num_frames=24_000)
    waveform, _ = load_audio(wav)
    assert waveform.shape == (1, TARGET_SAMPLES)


def test_load_audio_returns_float32(tmp_path):
    """Output dtype must be torch.float32 regardless of source encoding."""
    wav = _make_wav(tmp_path / "audio.wav", num_frames=24_000)
    waveform, _ = load_audio(wav)
    assert waveform.dtype == torch.float32


def test_load_audio_returns_target_sr(tmp_path):
    """Returned sample rate must equal target_sr."""
    wav = _make_wav(tmp_path / "audio.wav", num_frames=24_000)
    _, sr = load_audio(wav)
    assert sr == 16_000


# ---------------------------------------------------------------------------
# load_audio — input validation
# ---------------------------------------------------------------------------

def test_load_audio_wrong_type_raises_type_error():
    """Non-str file_path must raise TypeError immediately."""
    with pytest.raises(TypeError, match="file_path must be a str"):
        load_audio(123)  # type: ignore[arg-type]


def test_load_audio_missing_file_raises_file_not_found():
    """A path that does not exist must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_audio("/tmp/does_not_exist_xyz.wav")


# ---------------------------------------------------------------------------
# load_audio — resampling and channel conversion
# ---------------------------------------------------------------------------

def test_load_audio_resamples_44100_to_16000(tmp_path):
    """A 44 100 Hz source file must be downsampled to 16 000 Hz."""
    wav = _make_wav(tmp_path / "hifi.wav", num_frames=44_100, sr=44_100)
    waveform, sr = load_audio(wav)
    assert sr == 16_000
    # After resampling + fixed-duration enforcement, shape is always (1, 160000)
    assert waveform.shape == (1, TARGET_SAMPLES)


def test_load_audio_stereo_converts_to_mono(tmp_path):
    """A 2-channel source must be merged into a single channel."""
    wav = _make_wav(tmp_path / "stereo.wav", num_frames=24_000, channels=2)
    waveform, _ = load_audio(wav)
    assert waveform.shape[0] == 1


# ---------------------------------------------------------------------------
# normalize_waveform
# ---------------------------------------------------------------------------

def test_normalize_peak_max_is_one():
    """After peak normalization the absolute maximum must equal 1.0 (within fp tolerance)."""
    raw = torch.randn(1, 8_000) * 0.3
    normed = normalize_waveform(raw, method="peak")
    assert torch.isclose(normed.abs().max(), torch.tensor(1.0), atol=1e-6)


def test_normalize_rms_reduces_energy():
    """RMS normalization must return a tensor without extreme values."""
    raw = torch.randn(1, 8_000) * 10.0
    normed = normalize_waveform(raw, method="rms")
    # RMS of the output should be close to 1.0
    rms_out = normed.pow(2).mean().sqrt()
    assert torch.isclose(rms_out, torch.tensor(1.0), atol=1e-5)


def test_normalize_invalid_method_raises():
    """An unrecognised method name must raise ValueError."""
    t = torch.randn(1, 4_000)
    with pytest.raises(ValueError, match="method must be"):
        normalize_waveform(t, method="loudness")


def test_normalize_wrong_type_raises():
    """Passing a non-Tensor must raise TypeError."""
    with pytest.raises(TypeError, match="torch.Tensor"):
        normalize_waveform([0.1, 0.2, 0.3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# trim_silence
# ---------------------------------------------------------------------------

def test_trim_silence_shortens_waveform():
    """A waveform surrounded by zeros must be shorter after trimming."""
    signal = torch.cat([
        torch.zeros(1, 4_000),      # leading silence
        torch.ones(1, 4_000) * 0.8, # active signal
        torch.zeros(1, 4_000),      # trailing silence
    ], dim=-1)
    trimmed = trim_silence(signal, sr=16_000, top_db=30)
    assert trimmed.shape[-1] < signal.shape[-1]


# ---------------------------------------------------------------------------
# pad_or_truncate
# ---------------------------------------------------------------------------

def test_pad_or_truncate_right_pads_short_waveform():
    """A waveform shorter than the target must be right-padded to exact length."""
    short = torch.randn(1, 8_000)
    result = pad_or_truncate(short, sr=16_000, max_duration_sec=10.0)
    assert result.shape[-1] == TARGET_SAMPLES
    # Padding is on the right — the last samples must be zero
    assert result[0, 8_000:].eq(0).all()


def test_pad_or_truncate_center_crops_long_waveform():
    """A waveform longer than the target must be center-cropped to exact length."""
    long = torch.randn(1, 200_000)
    result = pad_or_truncate(long, sr=16_000, max_duration_sec=10.0)
    assert result.shape[-1] == TARGET_SAMPLES


def test_pad_or_truncate_exact_length_is_noop():
    """A waveform that is already the target length must be returned unchanged."""
    exact = torch.randn(1, TARGET_SAMPLES)
    result = pad_or_truncate(exact, sr=16_000, max_duration_sec=10.0)
    assert result.shape[-1] == TARGET_SAMPLES
    assert torch.equal(result, exact)


# ---------------------------------------------------------------------------
# load_audio — training flag enables silence trimming
# ---------------------------------------------------------------------------

def test_load_audio_training_flag_still_returns_correct_shape(tmp_path):
    """Even with training=True (silence trimming active) output is (1, 160000)."""
    wav = _make_wav(tmp_path / "train.wav", num_frames=24_000)
    waveform, _ = load_audio(wav, training=True)
    assert waveform.shape == (1, TARGET_SAMPLES)
