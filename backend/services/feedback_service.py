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
import io
import json
import logging
import os
import threading
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.labels import CLASS_LABELS

logger = logging.getLogger(__name__)

# Default cap, mirroring the mobile side (MAX_FEEDBACK_AUDIO_BYTES = 5 MB). The
# effective cap is configurable per-store (see FeedbackStore.__init__).
DEFAULT_MAX_AUDIO_BYTES = 5 * 1024 * 1024

# Valid stuttering classes — corrections outside this set would poison the
# retraining corpus, so they are rejected at the door.
VALID_LABELS = frozenset(str(label) for label in CLASS_LABELS)


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
    """Append-only store for HITL corrections (audio + a JSONL manifest).

    The manifest (``feedback.jsonl``) is the source of truth; the retraining
    reader should ignore any audio file not referenced by a manifest line.
    Intended for a single writer host — ``save`` is blocking and must be called
    off the event loop (the API offloads it to the threadpool).
    """

    def __init__(
        self, base_dir: str, max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES
    ) -> None:
        self.base_dir = Path(base_dir)
        self.audio_dir = self.base_dir / "audio"
        self.manifest_path = self.base_dir / "feedback.jsonl"
        self.max_audio_bytes = int(max_audio_bytes)
        # base64 expands bytes by ~4/3 — derive the max encoded length so we can
        # reject oversized payloads BEFORE the costly decode. Small margin for
        # any incidental whitespace/newlines.
        self._max_b64_len = 4 * ((self.max_audio_bytes + 2) // 3) + 16
        # Serialize concurrent writers appending to the same manifest file.
        self._lock = threading.Lock()
        # In-memory counter, seeded once, so /feedback doesn't re-scan the whole
        # manifest on every POST (the file grows unbounded over time).
        self._count = self._count_lines()

    def _count_lines(self) -> int:
        if not self.manifest_path.exists():
            return 0
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

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

        Raises ``FeedbackError`` on invalid/oversized audio, non-WAV payloads,
        empty/unknown ``correct_labels``, or a failed disk write.
        """
        # Reject oversized payloads BEFORE decoding, so a huge body can't force
        # a multi-MB base64 decode in memory just to be thrown away.
        if len(audio_base64) > self._max_b64_len:
            raise FeedbackError(
                f"audio exceeds {self.max_audio_bytes} bytes (encoded payload too large)"
            )
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise FeedbackError(f"audio_base64 is not valid base64: {exc}") from exc
        if not audio_bytes:
            raise FeedbackError("audio_base64 decoded to empty bytes")
        if len(audio_bytes) > self.max_audio_bytes:
            raise FeedbackError(
                f"audio exceeds {self.max_audio_bytes} bytes (got {len(audio_bytes)})"
            )
        # The payload is stored as *.wav and later read for retraining, so make
        # sure it really is decodable WAV audio — not arbitrary bytes.
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                if wav_file.getnchannels() <= 0 or wav_file.getframerate() <= 0:
                    raise FeedbackError("audio is not a valid WAV file")
        except (wave.Error, EOFError) as exc:
            # wave raises wave.Error for a wrong header but EOFError for data
            # too short to even hold a chunk header — both mean "not a WAV".
            raise FeedbackError(f"audio is not a valid WAV file: {exc}") from exc

        # correct_labels feed the retraining corpus — reject empty/unknown ones.
        norm_correct = _normalize_labels(correct_labels)
        if not norm_correct:
            raise FeedbackError("correct_labels must contain at least one label")
        unknown = sorted(set(norm_correct) - VALID_LABELS)
        if unknown:
            raise FeedbackError(f"unknown correct_labels: {', '.join(unknown)}")
        # original_prediction is only contextual metadata (what the AI said), so
        # an out-of-taxonomy value there must NOT discard a valid correction —
        # drop the unknowns and keep the rest.
        norm_original = _normalize_labels(original_prediction)
        dropped = [x for x in norm_original if x not in VALID_LABELS]
        if dropped:
            logger.warning(
                "Dropping unknown original_prediction labels: %s", ", ".join(dropped)
            )
            norm_original = [x for x in norm_original if x in VALID_LABELS]

        feedback_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        audio_name = f"{feedback_id}.wav"

        record = {
            "id": feedback_id,
            "created_at": created_at,
            "audio_file": f"audio/{audio_name}",
            "correct_labels": norm_correct,
            "original_prediction": norm_original,
            "model_version": str(model_version),
            "audio_bytes": len(audio_bytes),
        }

        self._ensure_dirs()
        audio_path = self.audio_dir / audio_name
        # Write the audio, then the manifest line (the manifest is the source of
        # truth: a manifest entry without committed audio must never exist). On
        # any I/O failure, remove the half-written audio so it can't be picked
        # up by the retraining reader. fsync the manifest so the line survives a
        # power loss. A hard kill between the two writes can orphan an audio
        # file — that's tolerated, since the reader ignores unreferenced audio.
        try:
            with self._lock:
                audio_path.write_bytes(audio_bytes)
                with self.manifest_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                self._count += 1
        except OSError as exc:
            audio_path.unlink(missing_ok=True)
            raise FeedbackError(f"failed to persist feedback: {exc}") from exc

        logger.info(
            "Stored HITL feedback %s (%d bytes, correct=%s, was=%s)",
            feedback_id,
            len(audio_bytes),
            record["correct_labels"],
            record["original_prediction"],
        )
        return record

    def count(self) -> int:
        """Number of corrections stored so far (O(1), in-memory counter)."""
        return self._count
