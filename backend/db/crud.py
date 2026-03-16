from typing import List
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import select
from .models import Prediction 
from .schemas import PredictionCreate
import logging
import uuid


logger = logging.getLogger(__name__)

async def create_prediction(db: AsyncSession, data: PredictionCreate) -> Prediction:
    values = data.dict()

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

    