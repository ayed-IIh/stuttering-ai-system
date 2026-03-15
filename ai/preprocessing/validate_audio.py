"""
Audio File Validation & Integrity Check — stuttering-ai-system
==============================================================
Loads every .wav listed in dataset_inventory.csv using both soundfile
and librosa as independent decoders.  Per-file checks:

  • decodability   — both libraries must load without exception          (FAIL)
  • sample rate    — must be 16 000 Hz                                   (FAIL)
  • channels       — flag stereo (> 1 channel)                          (WARN)
  • min duration   — reject clips shorter than 0.5 s                    (FAIL)
  • max duration   — flag clips longer than 30 s for review             (WARN)
  • RMS energy     — flag near-silent recordings below threshold         (FAIL)

Outputs:
  ai/preprocessing/validation_report.json  — structured per-file results
  ai/preprocessing/validate_audio.log      — full run log

Usage:
  python ai/preprocessing/validate_audio.py [OPTIONS]

  --inventory-csv PATH   default: ai/dataset/metadata/dataset_inventory.csv
  --output-dir    PATH   default: ai/preprocessing
  --rms-threshold FLOAT  default: 1e-4
  --log-level     LEVEL  default: INFO  {DEBUG, INFO, WARNING}

Exit:  0 if every file passes, 1 if any file fails.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

SCRIPT_VERSION = "1.0.0"

# ── Data models ────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ValidationConfig:
    """Single source of truth for all validation thresholds (OCP)."""

    expected_sample_rate: int = 16_000
    min_duration_seconds: float = 0.5
    max_duration_seconds: float = 30.0
    rms_silence_threshold: float = 1e-4


@dataclasses.dataclass
class AudioLoadResult:
    """Outcome of loading a file with one specific library."""

    library: str  # "torchaudio" | "librosa"
    success: bool
    error: Optional[str]
    sample_rate: Optional[int]
    channels: Optional[int]
    duration_seconds: Optional[float]
    # Raw numpy array for downstream RMS check — never stored in ValidationResult
    audio_array: Optional[np.ndarray]


@dataclasses.dataclass
class ValidationResult:
    """Per-file validation outcome committed to the JSON report."""

    filename: str
    absolute_path: str
    class_label: str
    passed: bool  # True iff len(failures) == 0
    failures: list = dataclasses.field(default_factory=list)
    warnings: list = dataclasses.field(default_factory=list)
    observed_sample_rate: Optional[int] = None
    observed_channels: Optional[int] = None
    observed_duration_seconds: Optional[float] = None
    torchaudio_ok: bool = False
    librosa_ok: bool = False
    torchaudio_error: Optional[str] = None
    librosa_error: Optional[str] = None


# ── Shared path utilities  ─────────────────────────────────────────────────────


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _to_abs_path(raw: str, *, must_exist: bool = False) -> Path:
    """Resolve path; relative paths anchored to project root, not cwd."""
    p = (Path(raw) if Path(raw).is_absolute() else _project_root() / raw).resolve()
    if must_exist and not p.exists():
        sys.exit(f"[ERROR] Path not found: {p}")
    return p


# ── Logging ────────────────────────────────────────────────────────────────────


def setup_logging(level: str, log_file: Path) -> logging.Logger:
    log = logging.getLogger("validate_audio")
    log.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


# ── Audio loaders (SRP: one library each) ─────────────────────────────────────


def load_with_torchaudio(path: Path) -> AudioLoadResult:
    """Attempt to decode *path* with torchaudio; never raises."""
    try:
        waveform, sr = torchaudio.load(str(path))
        audio_np = waveform.numpy()  # shape: (channels, samples)
        channels = audio_np.shape[0]
        duration = round(audio_np.shape[1] / sr, 6)
        return AudioLoadResult(
            library="torchaudio",
            success=True,
            error=None,
            sample_rate=int(sr),
            channels=channels,
            duration_seconds=duration,
            audio_array=audio_np,
        )
    except Exception as exc:  # noqa: BLE001
        return AudioLoadResult(
            library="torchaudio",
            success=False,
            error=str(exc),
            sample_rate=None,
            channels=None,
            duration_seconds=None,
            audio_array=None,
        )


def load_with_librosa(path: Path) -> AudioLoadResult:
    """Attempt to decode *path* with librosa; never raises."""
    try:
        y, sr = librosa.load(str(path), sr=None, mono=False)
        # librosa: mono → (samples,), multi → (channels, samples)
        if y.ndim == 1:
            channels, n_samples = 1, len(y)
        else:
            channels, n_samples = y.shape[0], y.shape[1]
        duration = round(n_samples / sr, 6)
        return AudioLoadResult(
            library="librosa",
            success=True,
            error=None,
            sample_rate=int(sr),
            channels=channels,
            duration_seconds=duration,
            audio_array=y,
        )
    except Exception as exc:  # noqa: BLE001
        return AudioLoadResult(
            library="librosa",
            success=False,
            error=str(exc),
            sample_rate=None,
            channels=None,
            duration_seconds=None,
            audio_array=None,
        )


# ── Check functions (SRP: one rule each, returns failure message or None) ──────


def check_sample_rate(sample_rate: int, expected: int) -> Optional[str]:
    """FAIL if sample_rate differs from expected."""
    if sample_rate != expected:
        return f"wrong_sample_rate: expected {expected}, got {sample_rate}"
    return None


def check_channels(channels: int) -> Optional[str]:
    """WARN if file is stereo (channels > 1)."""
    if channels > 1:
        return f"stereo_audio: {channels} channels detected, expected mono"
    return None


def check_duration_min(duration: float, min_seconds: float) -> Optional[str]:
    """FAIL if clip is shorter than minimum usable duration."""
    if duration < min_seconds:
        return f"too_short: {duration:.3f}s < minimum {min_seconds}s"
    return None


def check_duration_max(duration: float, max_seconds: float) -> Optional[str]:
    """WARN if clip exceeds maximum expected duration (flag for review)."""
    if duration > max_seconds:
        return f"too_long: {duration:.3f}s > {max_seconds}s, flag for review"
    return None


def check_rms_energy(audio_array: np.ndarray, threshold: float) -> Optional[str]:
    """FAIL if RMS energy falls below silence threshold."""
    rms = float(np.sqrt(np.mean(audio_array**2)))
    if rms < threshold:
        return f"silent_audio: rms={rms:.2e} < threshold {threshold:.2e}"
    return None


# ── Per-file orchestration ─────────────────────────────────────────────────────


def validate_file(
    path: Path,
    config: ValidationConfig,
    filename: str,
    class_label: str,
) -> ValidationResult:
    """Run all checks for one file; never raises.

    Accepts scalar ``filename`` and ``class_label`` so this function never
    reaches into a row-dict (Law of Demeter — the caller unpacks the row).
    """
    log = logging.getLogger("validate_audio")

    ta = load_with_torchaudio(path)
    lb = load_with_librosa(path)

    log.debug("torchaudio=%s librosa=%s  file=%s", ta.success, lb.success, filename)

    failures: list[str] = []
    warnings: list[str] = []

    if not ta.success:
        failures.append(f"torchaudio_decode_error: {ta.error}")
    if not lb.success:
        failures.append(f"librosa_decode_error: {lb.error}")

    # Prefer librosa values (already numpy-native); fall back to torchaudio
    sample_rate = lb.sample_rate or ta.sample_rate
    channels = lb.channels or ta.channels
    duration = lb.duration_seconds or ta.duration_seconds

    if sample_rate is not None:
        msg = check_sample_rate(sample_rate, config.expected_sample_rate)
        if msg:
            failures.append(msg)

    if channels is not None:
        msg = check_channels(channels)
        if msg:
            warnings.append(msg)

    if duration is not None:
        msg = check_duration_min(duration, config.min_duration_seconds)
        if msg:
            failures.append(msg)

        msg = check_duration_max(duration, config.max_duration_seconds)
        if msg:
            warnings.append(msg)

    if lb.audio_array is not None:
        msg = check_rms_energy(lb.audio_array, config.rms_silence_threshold)
        if msg:
            failures.append(msg)

    return ValidationResult(
        filename=filename,
        absolute_path=str(path),
        class_label=class_label,
        passed=len(failures) == 0,
        failures=failures,
        warnings=warnings,
        observed_sample_rate=sample_rate,
        observed_channels=channels,
        observed_duration_seconds=duration,
        torchaudio_ok=ta.success,
        librosa_ok=lb.success,
        torchaudio_error=ta.error,
        librosa_error=lb.error,
    )


# ── I/O helpers ────────────────────────────────────────────────────────────────


def load_inventory(csv_path: Path) -> list[dict]:
    """Read dataset_inventory.csv and return rows as plain dicts."""
    return pd.read_csv(csv_path).to_dict(orient="records")


def build_report(
    results: list[ValidationResult],
    ts: str,
    csv_path: Path,
    config: ValidationConfig,
) -> dict:
    """Assemble the JSON-serialisable report dict from validated results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    warned = sum(1 for r in results if r.warnings)
    pass_rate = round(passed / total * 100, 1) if total else 0.0

    return {
        "script_version": SCRIPT_VERSION,
        "generated_at": ts,
        "inventory_csv": str(csv_path),
        "config": dataclasses.asdict(config),
        "summary": {
            "total_files": total,
            "passed": passed,
            "failed": failed,
            "warned": warned,
            "pass_rate_pct": pass_rate,
        },
        "files": [dataclasses.asdict(r) for r in results],
    }


def write_report(report: dict, output_path: Path) -> None:
    """Write report as indented JSON — overwrites unconditionally (idempotent)."""
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_summary(results: list[ValidationResult]) -> None:
    """Print a human-readable summary to stdout."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    warned = sum(1 for r in results if r.warnings)
    pass_rate = passed / total * 100 if total else 0.0

    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  validate_audio v{SCRIPT_VERSION}  —  {total} files processed")
    print(f"  PASS : {passed:>4}  ({pass_rate:.1f}%)")
    print(f"  FAIL : {failed:>4}")
    print(f"  WARN : {warned:>4}  (warnings do not affect pass/fail)")
    print(f"{bar}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--inventory-csv",
        default="ai/dataset/metadata/dataset_inventory.csv",
        help="Path to dataset_inventory.csv  (default: ai/dataset/metadata/dataset_inventory.csv)",
    )
    p.add_argument(
        "--output-dir",
        default="ai/preprocessing",
        help="Directory for report and log  (default: ai/preprocessing)",
    )
    p.add_argument(
        "--rms-threshold",
        type=float,
        default=1e-4,
        help="RMS energy floor for silence detection  (default: 1e-4)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return p.parse_args()


# ── Orchestration ──────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    csv_path = _to_abs_path(args.inventory_csv, must_exist=True)
    out_dir = _to_abs_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(args.log_level, out_dir / "validate_audio.log")
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    config = ValidationConfig(rms_silence_threshold=args.rms_threshold)

    log.info("validate_audio v%s | %s | csv=%s", SCRIPT_VERSION, ts, csv_path)
    log.info(
        "Config: sample_rate=%d  min_dur=%.2fs  max_dur=%.0fs  rms_threshold=%.0e",
        config.expected_sample_rate,
        config.min_duration_seconds,
        config.max_duration_seconds,
        config.rms_silence_threshold,
    )

    rows = load_inventory(csv_path)
    log.info("Loaded %d rows from inventory CSV", len(rows))

    results: list[ValidationResult] = []
    for row in tqdm(rows, desc="Validating", unit="file", ncols=80):
        result = validate_file(
            path=Path(row["absolute_path"]),
            config=config,
            filename=row["filename"],
            class_label=row["class_label"],
        )
        results.append(result)

        if result.failures:
            log.warning("FAIL %s: %s", result.filename, "; ".join(result.failures))
        elif result.warnings:
            log.debug("WARN %s: %s", result.filename, "; ".join(result.warnings))

    report = build_report(results, ts, csv_path, config)
    report_path = out_dir / "validation_report.json"
    write_report(report, report_path)
    log.info("Report written → %s", report_path)

    print_summary(results)
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
