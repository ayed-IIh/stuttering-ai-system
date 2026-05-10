"""SQLAlchemy ORM models for the multi-label predictions schema.

Two-table layout:
    ``predictions``         — parent row, one per API call. Holds metadata,
                              ``all_scores`` JSONB (per-class sigmoid), and a
                              FK to ``model_versions``.
    ``prediction_classes``  — child row, one per detected class. Holds the
                              class label and its sigmoid confidence.

The legacy single-label ``predicted_class`` enum column was dropped in
``backend/db/migrations/002_multi_label_predictions.sql``.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .database import Base


class ModelVersion(Base):
    """A trained model artifact registered in the registry table."""

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    deployed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text)
    notes: Mapped[str] = mapped_column(Text)


class Prediction(Base):
    """One row per API call.

    ``all_scores`` is a JSONB dict with one entry per class in
    ``shared.labels.CLASS_LABELS``. Values are independent sigmoid
    probabilities and do NOT sum to 1.0. The per-class threshold filter
    that determines which classes appear in ``prediction_classes`` is
    applied server-side at inference time, not in the DB.
    """

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    audio_filename: Mapped[str] = mapped_column(String(500))
    audio_duration_sec: Mapped[float] = mapped_column(Float)
    all_scores: Mapped[dict] = mapped_column(JSONB, nullable=False)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    processing_time_ms: Mapped[int] = mapped_column(Integer)
    client_ip: Mapped[str] = mapped_column(INET)
    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), unique=True, nullable=False
    )

    # Cascade delete keeps the child table tidy on prediction row deletes.
    prediction_classes: Mapped[list["PredictionClass"]] = relationship(
        "PredictionClass",
        back_populates="prediction",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "jsonb_typeof(all_scores) = 'object'",
            name="chk_all_scores_type",
        ),
        Index("idx_predictions_created_at", "created_at"),
        Index("idx_predictions_model_version_id", "model_version_id"),
    )


class PredictionClass(Base):
    """One row per detected class for a given prediction.

    Populated by :func:`backend.api.routes._persist_multi_label_prediction`
    after the server-side threshold filter. ``confidence`` is the sigmoid
    probability for this specific class — not a row-relative weight.
    """

    __tablename__ = "prediction_classes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
    )
    class_label: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(6, 5), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    prediction: Mapped[Prediction] = relationship(
        "Prediction", back_populates="prediction_classes"
    )

    __table_args__ = (
        CheckConstraint(
            "confidence BETWEEN 0 AND 1",
            name="chk_prediction_classes_confidence_range",
        ),
        CheckConstraint(
            "class_label = ANY(ARRAY["
            "'fluent','blocks','interjections','prolongations',"
            "'part_word_repetition','phrase_repetition','word_repetition'"
            "])",
            name="chk_prediction_classes_class_label",
        ),
        UniqueConstraint(
            "prediction_id",
            "class_label",
            name="uq_prediction_classes_prediction_id_class_label",
        ),
        Index("idx_prediction_classes_prediction_id", "prediction_id"),
        Index("idx_prediction_classes_class_label", "class_label"),
    )
