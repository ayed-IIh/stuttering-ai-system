from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from shared.labels import CLASS_LABELS

VALID_FEEDBACK_LABELS: tuple[str, ...] = CLASS_LABELS


def feedback_samples_dir(base_dir: Path | None = None) -> Path:
    return (base_dir or Path.cwd()) / "feedback_samples"


def save_feedback_sample(
    *,
    audio_bytes: bytes,
    correct_labels: list[str],
    original_prediction: str,
    model_version: str,
    base_dir: Path | None = None,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    sample_id = uuid.uuid4().hex
    root = feedback_samples_dir(base_dir)
    metadata = {
        "timestamp": timestamp,
        "original_prediction": original_prediction,
        "correct_labels": correct_labels,
        "model_version": model_version,
    }

    for label in correct_labels:
        label_dir = root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{timestamp.replace(':', '-').replace('+', 'Z')}_{sample_id}"
        audio_path = label_dir / f"{stem}.wav"
        metadata_path = label_dir / f"{stem}.json"
        audio_path.write_bytes(audio_bytes)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
