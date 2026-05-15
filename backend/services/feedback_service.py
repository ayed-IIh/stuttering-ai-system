from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from shared.labels import CLASS_LABELS

VALID_FEEDBACK_LABELS: tuple[str, ...] = CLASS_LABELS


def feedback_samples_dir(base_dir: Path | None = None) -> Path:
    """
    Compute the root "feedback_samples" directory under the given base directory or the current working directory.
    
    Parameters:
        base_dir (Path | None): Optional base directory; when None, the current working directory is used.
    
    Returns:
        Path: Path to the "feedback_samples" directory (base_dir / "feedback_samples").
    """
    return (base_dir or Path.cwd()) / "feedback_samples"


def save_feedback_sample(
    *,
    audio_bytes: bytes,
    correct_labels: list[str],
    original_prediction: str,
    model_version: str,
    base_dir: Path | None = None,
) -> None:
    """
    Persist an audio sample and its metadata into label-specific subdirectories under the feedback_samples directory.
    
    Parameters:
        audio_bytes (bytes): Raw audio data to save as a .wav file.
        correct_labels (list[str]): Labels to associate with this sample; the function creates one .wav and one .json metadata file inside each corresponding label subdirectory.
        original_prediction (str): The model's original predicted label to record in the metadata.
        model_version (str): Identifier of the model version to record in the metadata.
        base_dir (Path | None): Optional root directory to use instead of the current working directory; the function stores files under `<base_dir or cwd>/feedback_samples`.
    
    """
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
