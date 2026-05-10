"""Pydantic v2 wire schemas for the multi-label predictions API.

These shapes pair with the ORM models in ``backend.db.models`` and the wire
contract in ``docs/api_contract.md`` v2.0. They are used internally by
``backend.db.crud`` and the route handlers — not exposed in the HTTP layer
directly.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress, field_validator

from shared.labels import CLASS_LABELS, NUM_CLASSES


class PredictionCreate(BaseModel):
    """Parent ``predictions`` row payload (one per API call)."""

    audio_filename: str
    audio_duration_sec: float
    all_scores: dict[str, float] = Field(
        ...,
        description=(
            "Per-class sigmoid probabilities keyed by shared.labels.CLASS_LABELS. "
            "Values do NOT sum to 1.0 — each is an independent sigmoid output."
        ),
    )
    model_version_id: UUID
    processing_time_ms: int
    client_ip: IPvAnyAddress
    request_id: Optional[UUID] = None

    @field_validator("all_scores")
    @classmethod
    def _all_scores_match_class_labels(
        cls, v: dict[str, float]
    ) -> dict[str, float]:
        """Reject payloads whose keys don't exactly match CLASS_LABELS."""
        expected = set(CLASS_LABELS)
        actual = set(v.keys())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "all_scores keys must equal CLASS_LABELS; "
                f"missing={missing} extra={extra}"
            )
        if len(v) != NUM_CLASSES:
            raise ValueError(
                f"all_scores must have exactly {NUM_CLASSES} entries; got {len(v)}"
            )
        for name, score in v.items():
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError(
                    f"all_scores[{name!r}]={score!r} out of [0, 1]"
                )
        return v


class PredictionClassCreate(BaseModel):
    """One child row in ``prediction_classes``.

    Used by :func:`backend.db.crud.insert_prediction_classes` to validate
    callers' per-class payloads before issuing the INSERT.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prediction_id: UUID
    class_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("class_label")
    @classmethod
    def _class_label_in_taxonomy(cls, v: str) -> str:
        """class_label must be one of the seven canonical labels."""
        if v not in CLASS_LABELS:
            raise ValueError(
                f"class_label must be one of {list(CLASS_LABELS)}; got {v!r}"
            )
        return v
