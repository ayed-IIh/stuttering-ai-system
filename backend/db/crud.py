"""Database CRUD helpers for the multi-label predictions schema.

Two-table writes:
    1. :func:`create_prediction` inserts the parent ``predictions`` row.
    2. :func:`insert_prediction_classes` inserts 0..N child
       ``prediction_classes`` rows after the threshold filter.

Use the two together when persisting a complete prediction; the route
handler in ``backend/api/routes.py`` does this.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ModelVersion, Prediction, PredictionClass
from .schemas import PredictionClassCreate, PredictionCreate

logger = logging.getLogger(__name__)


async def get_model_version_id_by_label(
    db: AsyncSession, model_version: str
) -> Optional[uuid.UUID]:
    """Resolve ``model_versions.id`` from the version label.

    Args:
        db: Open async DB session.
        model_version: Version string from the loaded checkpoint (e.g. ``"1.0.0"``).

    Returns:
        The matching ``ModelVersion.id`` or ``None`` when nothing matches or
        the lookup itself errors. Errors are logged but never re-raised so
        the caller can decide whether to skip persistence.
    """
    try:
        result = await db.execute(
            select(ModelVersion.id)
            .where(ModelVersion.model_version == model_version)
            .order_by(ModelVersion.deployed_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
    except SQLAlchemyError as exc:
        logger.warning(
            "Could not resolve model_version_id for %r: %s", model_version, exc
        )
        return None


async def create_prediction(
    db: AsyncSession, data: PredictionCreate
) -> Prediction:
    """Insert one parent ``predictions`` row.

    Args:
        db: Open async DB session.
        data: Validated :class:`PredictionCreate` payload.

    Returns:
        The committed :class:`Prediction` ORM instance.

    Raises:
        HTTPException: 400 on DB constraint violations, 500 on other DB errors.
    """
    values = data.model_dump()

    if not values.get("request_id"):
        values["request_id"] = uuid.uuid4()

    prediction = Prediction(**values)

    try:
        db.add(prediction)
        await db.commit()
        await db.refresh(prediction)
        return prediction

    except IntegrityError as exc:
        await db.rollback()
        msg = str(exc.orig) if hasattr(exc, "orig") else str(exc)

        if "chk_all_scores_type" in msg:
            detail = "all_scores must be a JSON object."
        elif "fk_predictions_model_version_id" in msg or "model_version_id" in msg:
            detail = "model_version_id does not exist. Create/find a model version first."
        else:
            detail = "Payload violates DB constraints."

        raise HTTPException(status_code=400, detail=detail)

    except SQLAlchemyError as exc:
        await db.rollback()
        logger.error("Database error while creating prediction: %s", exc)
        raise HTTPException(status_code=500, detail="Internal DB error.")

    except Exception:
        await db.rollback()
        logger.exception("Unexpected error while creating prediction")
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while creating prediction",
        )


async def insert_prediction_classes(
    db: AsyncSession,
    prediction_id: uuid.UUID,
    classes: List[dict],
) -> int:
    """Insert one ``prediction_classes`` row per detected class.

    Args:
        db: Open async DB session. The caller is responsible for the parent
            ``predictions`` row already existing — the FK on
            ``prediction_classes.prediction_id`` will reject orphan inserts.
        prediction_id: UUID of the parent row.
        classes: List of ``{"class_label": str, "confidence": float}`` dicts.
            Each entry is validated against :class:`PredictionClassCreate`
            (which enforces taxonomy + range) before insert.

    Returns:
        Number of rows inserted. Returns 0 (and logs a warning) on any
        :class:`SQLAlchemyError` during the write — this function never
        raises a DB error to the caller. ``ValueError`` from the per-entry
        validator IS raised so callers can surface bad payloads at the API
        boundary.

    Raises:
        ValueError: If any entry fails :class:`PredictionClassCreate` validation.
    """
    if not classes:
        return 0

    rows: list[PredictionClass] = []
    for entry in classes:
        payload = PredictionClassCreate(
            prediction_id=prediction_id,
            class_label=entry["class_label"],
            confidence=float(entry["confidence"]),
        )
        rows.append(
            PredictionClass(
                prediction_id=payload.prediction_id,
                class_label=payload.class_label,
                confidence=payload.confidence,
            )
        )

    try:
        db.add_all(rows)
        await db.commit()
        return len(rows)
    except SQLAlchemyError as exc:
        logger.warning(
            "insert_prediction_classes failed for %s: %s", prediction_id, exc
        )
        try:
            await db.rollback()
        except SQLAlchemyError:
            logger.exception("rollback also failed for %s", prediction_id)
        return 0


async def get_recent_predictions(
    db: AsyncSession, limit: int = 50
) -> List[Prediction]:
    """Fetch the most recent predictions, newest first."""
    limit = max(1, min(limit, 50))
    try:
        result = await db.execute(
            select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
        )
        predictions = result.scalars().all()
        return predictions
    except SQLAlchemyError as exc:
        logger.error("Database error while fetching predictions: %s", exc)
        raise HTTPException(status_code=500, detail="Internal DB error.")
