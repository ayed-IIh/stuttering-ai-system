from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy import (
    String, Text, DateTime,
    Float, Boolean, Integer,
    ForeignKey, CheckConstraint,
    Index, Enum as SQLAlchemyEnum
)
from sqlalchemy.sql import func
from .database import Base
import uuid
import enum


# Enum for predicted_class
class StutterClass(str, enum.Enum):
    fluent = "fluent"
    blocks = "blocks"
    interjections = "interjections"
    prolongations = "prolongations"
    part_word_repetition = "part_word_repetition"
    phrase_repetition = "phrase_repetition"
    word_repetition = "word_repetition"


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    deployed_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text)
    notes: Mapped[str] = mapped_column(Text)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    audio_filename: Mapped[str] = mapped_column(String(500))
    audio_duration_sec: Mapped[float] = mapped_column(Float)
    predicted_class: Mapped[StutterClass] = mapped_column(
        SQLAlchemyEnum(StutterClass, name="stutterclass", create_type=True),
        nullable=False
    )
    confidence_scores: Mapped[dict] = mapped_column(JSONB, nullable=False)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id", ondelete="RESTRICT"),
        nullable=False
    )
    processing_time_ms: Mapped[int] = mapped_column(Integer)
    client_ip: Mapped[str] = mapped_column(INET)
    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), unique=True, nullable=False
    )

    # Table-level constraints & indexes
    __table_args__ = (
        # Optional: simple JSONB type check (advanced key/value validation still in DB migration)
        CheckConstraint(
            "jsonb_typeof(confidence_scores) = 'object'",
            name="chk_confidence_score_type"
        ),
        # Indexes
        Index("idx_predictions_created_at", "created_at"),
        Index("idx_predictions_predicted_class", "predicted_class"),
        Index("idx_predictions_model_version_id", "model_version_id"),
    )
