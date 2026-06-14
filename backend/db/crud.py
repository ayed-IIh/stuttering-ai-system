from typing import List, Optional

import logging
import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ModelVersion, Prediction
from .schemas import PredictionCreate

logger = logging.getLogger(__name__)


async def get_model_version_id_by_label(
    db: AsyncSession, model_version: str
) -> Optional[uuid.UUID]:
    """Resolve ``model_versions.id`` from the string label (e.g. ``0.1.0``)."""
    try:
        result = await db.execute(
            select(ModelVersion.id)
            .where(ModelVersion.model_version == model_version)
            .order_by(ModelVersion.deployed_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
    except SQLAlchemyError as e:
        logger.warning("Could not resolve model_version_id for %r: %s", model_version, e)
        return None


async def create_prediction(db: AsyncSession, data: PredictionCreate) -> Prediction:
    values = data.model_dump()

    if not values.get("request_id"):
        values["request_id"] = uuid.uuid4()

    prediction = Prediction(**values)

    try:
        db.add(prediction)
        await db.commit()
        await db.refresh(prediction)
        return prediction

    except IntegrityError as e:
        await db.rollback()
        msg = str(e.orig) if hasattr(e, "orig") else str(e)

        if "chk_predicted_class" in msg:
            detail = "predicted_class must be one of the allowed classes."
        elif "chk_confidence_scores_keys" in msg:
            detail = "confidence_scores must contain exactly the 7 class keys."
        elif "fk_predictions_model_version_id" in msg or "model_version_id" in msg:
            detail = "model_version_id does not exist. Create/find a model version first."
        else:
            detail = "Payload violates DB constraints."

        raise HTTPException(status_code=400, detail=detail)

    except SQLAlchemyError as e:
        await db.rollback()
        logger.error("Database error while creating prediction: %s", e)
        raise HTTPException(status_code=500, detail="Internal DB error.")

    except Exception:
        await db.rollback()
        logger.exception("Unexpected error while creating prediction")
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while creating prediction",
        )


async def get_recent_predictions(db: AsyncSession, limit: int = 50) -> List[Prediction]:
    limit = max(1, min(limit, 50))
    try:
        result = await db.execute(
            select(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
        )
        predictions = result.scalars().all()
        return predictions
    except SQLAlchemyError as e:
        logger.error("Database error while fetching predictions: %s", e)
        raise HTTPException(status_code=500, detail="Internal DB error.")
