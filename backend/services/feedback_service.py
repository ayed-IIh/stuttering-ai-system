"""Persist therapist corrections (HITL feedback) for later retraining.

The mobile client POSTs corrections to ``/api/v1/feedback`` whenever a
therapist edits the AI's diagnosis. Each correction is stored as:

  - the WAV audio (decoded from base64) under ``<FEEDBACK_DIR>/audio/<id>.wav``
  - a JSON line appended to ``<FEEDBACK_DIR>/feedback.jsonl``

A future retraining run reads ``feedback.jsonl`` plus the audio files to extend
the training manifest with real, therapist-verified labels — this is the data
lever that lifts accuracy on the rare classes over time.

The store is intentionally file-based (no DB dependency) so HITL data survives
regardless of whether Postgres is configured.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mirror the mobile-side cap (MAX_FEEDBACK_AUDIO_BYTES = 5 MB) so we reject
# oversized payloads rather than writing them to disk.
MAX_AUDIO_BYTES = 5 * 1024 * 1024


class FeedbackError(Exception):
    """Raised when a feedback payload cannot be stored (bad/oversized audio)."""


def _normalize_labels(value: Any) -> list[str]:
    """Coerce a label collection into a flat list of label strings.

    Accepts a list of strings, a list of ``{"class": ...}`` / ``{"label": ...}``
    objects (the shape the mobile re-extracts from a stored prediction), a bare
    string, or ``None``.
    """
    out: list[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                label = item.get("class") or item.get("label")
                if label:
                    out.append(str(label))
            elif item is not None:
                out.append(str(item))
    return out


class FeedbackStore:
    """Append-only store for HITL corrections (audio + a JSONL manifest)."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.audio_dir = self.base_dir / "audio"
        self.manifest_path = self.base_dir / "feedback.jsonl"
        # Serialize concurrent writers appending to the same manifest file.
        self._lock = threading.Lock()

    def _ensure_dirs(self) -> None:
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        audio_base64: str,
        correct_labels: Any,
        original_prediction: Any,
        model_version: str,
    ) -> dict:
        """Decode + store one correction. Returns the stored manifest record.

        Raises ``FeedbackError`` on invalid or oversized audio.
        """
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise FeedbackError(f"audio_base64 is not valid base64: {exc}") from exc
        if not audio_bytes:
            raise FeedbackError("audio_base64 decoded to empty bytes")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise FeedbackError(
                f"audio exceeds {MAX_AUDIO_BYTES} bytes (got {len(audio_bytes)})"
            )

        feedback_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        audio_name = f"{feedback_id}.wav"

        record = {
            "id": feedback_id,
            "created_at": created_at,
            "audio_file": f"audio/{audio_name}",
            "correct_labels": _normalize_labels(correct_labels),
            "original_prediction": _normalize_labels(original_prediction),
            "model_version": str(model_version),
            "audio_bytes": len(audio_bytes),
        }

        self._ensure_dirs()
        with self._lock:
            (self.audio_dir / audio_name).write_bytes(audio_bytes)
            with self.manifest_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "Stored HITL feedback %s (%d bytes, correct=%s, was=%s)",
            feedback_id,
            len(audio_bytes),
            record["correct_labels"],
            record["original_prediction"],
        )
        return record

    def count(self) -> int:
        """Number of corrections stored so far (0 if none yet)."""
        if not self.manifest_path.exists():
            return 0
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
