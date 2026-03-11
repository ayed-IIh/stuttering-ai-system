"""
Dataset Audit Script — stuttering-ai-system
============================================
Walks the 7-class labeled .wav dataset and emits:
  - ai/dataset/metadata/dataset_inventory.csv
  - ai/dataset/metadata/inventory_report.md
  - ai/dataset/metadata/audit.log

Usage:  python scripts/audit_dataset.py [--dataset-root PATH] [--output-dir PATH] [--log-level LEVEL]
Exit:   0 if >=95% files valid, 1 otherwise
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

SCRIPT_VERSION = "1.0.0"
VALID_FILE_THRESHOLD = 1
VALID_EXTENSIONS = frozenset({".wav"})

# Directories inside dataset root to skip during discovery (OCP: extend here)
EXCLUDE_DIRS: frozenset[str] = frozenset({"_duplicates"})

# Single source of truth for taxonomy — extend here only (OCP)
TAXONOMY: dict[str, str] = {
    "Fluent":                              "fluent",
    "Blocks":                              "blocks",
    "Interjections":                       "interjections",
    "Prolongations":                       "prolongations",
    "Repetitions/Part-word repetition":    "part_word_repetition",
    "Repetitions/Phrase repetition":       "phrase_repetition",
    "Repetitions/Word repetition":         "word_repetition",
}

# Add entries here to extend filename validation — no code changes needed (OCP)
BAD_FILENAME_PATTERNS: list[re.Pattern] = [
    re.compile(r"^Copy of", re.IGNORECASE),
    re.compile(r"\s"),
    re.compile(r"[^a-zA-Z0-9._\-]"),
]

# DRY anchor: drives both compute_flags_summary() and the anomaly report
FLAG_FIELDS: list[tuple[str, str]] = [
    ("is_zero_byte",             "zero_byte"),
    ("is_corrupted",             "corrupted"),
    ("is_duplicate",             "duplicate"),
    ("has_missing_label",        "missing_label"),
    ("has_nonstandard_filename", "nonstandard_filename"),
]

CSV_COLUMNS: list[str] = [
    "filename", "absolute_path", "class_label",
    "file_size_bytes", "sample_rate_hz", "bit_depth", "channels", "duration_seconds", "md5_hash",
    "is_zero_byte", "is_corrupted", "is_duplicate", "has_missing_label", "has_nonstandard_filename",
    "flags_summary",
]

# ── Data model ─────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class FileRecord:
    filename: str
    absolute_path: str
    class_label: str
    file_size_bytes: int
    sample_rate_hz: Optional[int]
    bit_depth: Optional[int]
    channels: Optional[int]
    duration_seconds: Optional[float]
    md5_hash: Optional[str]
    is_zero_byte: bool = False
    is_corrupted: bool = False
    is_duplicate: bool = False
    has_missing_label: bool = False
    has_nonstandard_filename: bool = False
    flags_summary: str = ""

# ── Shared utilities ───────────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _to_abs_path(raw: str, *, must_exist: bool = False) -> Path:
    """Resolve path; relative paths anchored to project root, not cwd."""
    p = (Path(raw) if Path(raw).is_absolute() else _project_root() / raw).resolve()
    if must_exist and not p.exists():
        sys.exit(f"[ERROR] Path not found: {p}")
    return p

def _md_table(headers: list[str], rows: list) -> str:
    """DRY Markdown table builder — single implementation used everywhere."""
    sep = "|".join("---" for _ in headers)
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return f"| {' | '.join(headers)} |\n|{sep}|\n{body}"

# ── Core functions (SRP — one job each) ───────────────────────────────────────

def setup_logging(level: str, log_file: Path) -> logging.Logger:
    log = logging.getLogger("audit_dataset")
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log

def resolve_label(path: Path, root: Path) -> tuple[str, bool]:
    """Map folder path → taxonomy label. Returns (label, has_missing_label)."""
    try:
        key = "/".join(path.relative_to(root).parts[:-1])
    except ValueError:
        return "UNKNOWN", True
    label = TAXONOMY.get(key)
    return (label, False) if label else ("UNKNOWN", True)

def compute_md5(path: Path, chunk: int = 8192) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def parse_bit_depth(subtype: str) -> Optional[int]:
    """Parse soundfile subtype string → integer bit depth."""
    m = re.search(r"(\d+)$", subtype)
    return int(m.group(1)) if m else {"FLOAT": 32, "DOUBLE": 64, "VORBIS": 16}.get(subtype.upper())

def _read_audio_info(path: Path) -> Optional[dict]:
    """librosa for audio properties; soundfile for bit_depth only (SRP).
    Returns None if librosa cannot open the file (file is corrupted)."""
    log = logging.getLogger("audit_dataset")
    try:
        y, sr = librosa.load(str(path), sr=None, mono=False)
        channels = 1 if y.ndim == 1 else y.shape[0]
        n_samples = len(y) if y.ndim == 1 else y.shape[1]
    except Exception as e:
        log.warning("Corrupted: %s — %s", path.name, e)
        return None

    # soundfile for bit_depth only — failure is non-fatal
    bit_depth: Optional[int] = None
    try:
        with sf.SoundFile(path) as f:
            bit_depth = parse_bit_depth(f.subtype)
    except Exception:
        log.debug("bit_depth unavailable for %s", path.name)

    return {
        "sample_rate_hz": int(sr),
        "bit_depth": bit_depth,
        "channels": channels,
        "duration_seconds": round(n_samples / sr, 6),
    }

def compute_flags_summary(r: FileRecord) -> str:
    """DRY anchor — single source of truth for flag→label. Used by collect + detect."""
    return ", ".join(label for field, label in FLAG_FIELDS if getattr(r, field))

def collect_file_metadata(path: Path, root: Path) -> FileRecord:
    """Compose helpers into one FileRecord. is_duplicate resolved in second pass."""
    size = path.stat().st_size
    zero = size == 0
    label, missing = resolve_label(path, root)
    audio = None if zero else _read_audio_info(path)
    r = FileRecord(
        filename=path.name, absolute_path=str(path), class_label=label,
        file_size_bytes=size,
        sample_rate_hz=audio and audio["sample_rate_hz"],
        bit_depth=audio and audio["bit_depth"],
        channels=audio and audio["channels"],
        duration_seconds=audio and audio["duration_seconds"],
        md5_hash=compute_md5(path) if audio else None,
        is_zero_byte=zero,
        is_corrupted=not zero and audio is None,
        has_missing_label=missing,
        has_nonstandard_filename=any(p.search(path.name) for p in BAD_FILENAME_PATTERNS),
    )
    r.flags_summary = compute_flags_summary(r)
    return r

def detect_duplicates(records: list[FileRecord]) -> None:
    """Group by MD5, mark duplicates, refresh flags_summary in place."""
    log = logging.getLogger("audit_dataset")
    groups: dict[str, list[FileRecord]] = defaultdict(list)
    for r in records:
        if r.md5_hash:
            groups[r.md5_hash].append(r)
    for md5, grp in groups.items():
        if len(grp) < 2:
            continue
        labels = {r.class_label for r in grp}
        (log.warning if len(labels) > 1 else log.info)(
            "Duplicates (md5=%.8s, labels=%s): %s", md5, sorted(labels), [r.filename for r in grp]
        )
        for r in grp:
            r.is_duplicate = True
            r.flags_summary = compute_flags_summary(r)

def write_csv(records: list[FileRecord], path: Path) -> None:
    pd.DataFrame([dataclasses.asdict(r) for r in records], columns=CSV_COLUMNS).to_csv(path, index=False)

def write_report(records: list[FileRecord], path: Path, root: Path, ts: str) -> None:
    total = len(records)
    flagged = [r for r in records if r.flags_summary]
    valid_n = total - len(flagged)
    pct = valid_n / total * 100 if total else 0.0

    path.write_text("\n\n".join([
        f"# Dataset Audit Report\n\n- **v{SCRIPT_VERSION}** | {ts} | `{root}`",
        "## Summary\n\n" + _md_table(
            ["Metric", "Value"],
            [["Total", total], ["Valid", valid_n], ["Flagged", len(flagged)],
             ["Valid %", f"{pct:.1f}%"],
             ["Status", "PASS" if pct >= VALID_FILE_THRESHOLD * 100 else "❌ FAIL"]],
        ),
        "## File Count per Class\n\n" + _md_table(
            ["Class", "Count"], sorted(Counter(r.class_label for r in records).items())
        ),
        "## Format Distribution\n\n"
        + "**Sample Rate (Hz)**\n" + _md_table(["Rate", "Count"], sorted(Counter(r.sample_rate_hz for r in records).items())) + "\n\n"
        + "**Bit Depth**\n"        + _md_table(["Depth", "Count"], sorted(Counter(r.bit_depth for r in records).items())) + "\n\n"
        + "**Channels**\n"         + _md_table(["Channels", "Count"], sorted(Counter(r.channels for r in records).items())),
        "## Anomaly Breakdown\n\n" + _md_table(
            ["Flag", "Count"],
            [[lbl, sum(1 for r in records if getattr(r, fld))] for fld, lbl in FLAG_FIELDS],
        ),
        "## Flagged Files\n\n" + (
            _md_table(["Filename", "Class", "Flags"],
                      [[f"`{r.filename}`", r.class_label, r.flags_summary]
                       for r in sorted(flagged, key=lambda r: r.filename)])
            if flagged else "_None — all files clean._"
        ),
        "## Files Requiring Rename\n\n" + (
            _md_table(["Filename", "Class"],
                      [[f"`{r.filename}`", r.class_label]
                       for r in sorted(records, key=lambda r: r.filename)
                       if r.has_nonstandard_filename])
            if any(r.has_nonstandard_filename for r in records) else "_None — all filenames compliant._"
        ),
    ]) + "\n", encoding="utf-8")

# ── CLI + orchestration ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="ai/dataset/raw",      help="Dataset root (default: ai/dataset/raw)")
    p.add_argument("--output-dir",   default="ai/dataset/metadata", help="Output dir  (default: ai/dataset/metadata)")
    p.add_argument("--log-level",    default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()

def main() -> None:
    args = parse_args()
    root = _to_abs_path(args.dataset_root, must_exist=True)
    out  = _to_abs_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log = setup_logging(args.log_level, out / "audit.log")
    ts  = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log.info("audit_dataset v%s | %s | root=%s", SCRIPT_VERSION, ts, root)

    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VALID_EXTENSIONS
        and not any(part in EXCLUDE_DIRS for part in p.relative_to(root).parts)
    )
    if not paths:
        log.warning("No .wav files found under %s", root)
        write_csv([], out / "dataset_inventory.csv")
        write_report([], out / "inventory_report.md", root, ts)
        sys.exit(1)

    log.info("Found %d file(s). Collecting metadata...", len(paths))
    records = [collect_file_metadata(p, root) for p in tqdm(paths, desc="Auditing", unit="file", ncols=80)]

    detect_duplicates(records)
    write_csv(records, out / "dataset_inventory.csv")
    write_report(records, out / "inventory_report.md", root, ts)

    valid_n = sum(1 for r in records if not r.flags_summary)
    pct = valid_n / len(records) * 100
    log.info("Result: %d/%d valid (%.1f%%) — threshold %.0f%%", valid_n, len(records), pct, VALID_FILE_THRESHOLD * 100)
    sys.exit(0 if pct >= VALID_FILE_THRESHOLD * 100 else 1)

if __name__ == "__main__":
    main()
