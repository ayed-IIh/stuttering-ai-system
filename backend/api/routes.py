"""HTTP route handlers for the multi-label inference API.

POST /predict accepts a WAV upload and returns multi-label predictions:
each class whose sigmoid probability ``>= threshold`` is included in
``predicted_classes``. The full per-class sigmoid distribution is in
``all_scores`` (NOT softmax — values do not sum to 1.0).

DB persistence: enabled. Migration ``002_multi_label_predictions.sql`` adds
the ``prediction_classes`` child table; this module inserts one row per
detected class via :func:`_persist_multi_label_prediction`. Failures are
logged but never propagated to the client.
"""

from __future__ import annotations

import io
import logging
import time
import uuid
import wave
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.middleware import (
    RequestSizeLimitExceeded,
    validate_audio_upload,
)
from backend.db.database import get_db
from backend.services.model_service import (
    InvalidAudioError,
    ModelNotLoadedError,
    ModelService,
    PredictionError,
)
from shared.labels import CLASS_LABELS, ID2LABEL, LABEL2ID

logger = logging.getLogger(__name__)


class PredictedClass(BaseModel):
    """One detected class in a multi-label prediction."""

    model_config = ConfigDict(populate_by_name=True)
    class_name: str = Field(
        ..., alias="class", description="One of shared.labels.CLASS_LABELS"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Multi-label /predict response.

    Notes:
        * ``predicted_classes`` lists every class with sigmoid ≥ ``threshold``,
          sorted by descending confidence. May be empty if no class crosses.
        * ``all_scores`` always contains exactly the 7 keys from CLASS_LABELS.
          Values are independent sigmoid probabilities and do **not** sum to 1.0.
    """

    predicted_classes: list[PredictedClass]
    all_scores: dict[str, float]
    threshold: float
    processing_time_ms: int
    model_version: str
    request_id: str

    @field_validator("all_scores")
    @classmethod
    def _all_scores_match_class_labels(cls, v: dict[str, float]) -> dict[str, float]:
        """all_scores must have exactly the keys defined in CLASS_LABELS."""
        expected = set(CLASS_LABELS)
        actual = set(v.keys())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "all_scores keys must equal CLASS_LABELS; "
                f"missing={missing} extra={extra}"
            )
        for name, score in v.items():
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError(
                    f"all_scores[{name!r}]={score!r} out of [0, 1]"
                )
        return v


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: int | None = None


class ClassesResponse(BaseModel):
    classes: list[str]
    label_to_id: dict[str, int]
    id_to_label: dict[str, str]


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    detail: str


def get_model_service(request: Request) -> ModelService:
    """Resolve the singleton ModelService from app state or fail with 503."""
    model_service = getattr(request.app.state, "model_service", None)
    if model_service is None:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error_code="SERVICE_UNAVAILABLE",
                message="Service is not ready",
                detail="Model service is not initialized",
            ).model_dump(),
        )
    return model_service


router = APIRouter()

_PROCESS_START_WALL = time.time()


def _uptime_seconds() -> int:
    return int(time.time() - _PROCESS_START_WALL)


def _wav_duration_sec(audio_bytes: bytes) -> float:
    """Best-effort WAV duration parse — returns 0.0 if anything's off."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except wave.Error:
        return 0.0
    except EOFError:
        return 0.0


def _build_prediction_response(
    prediction: dict[str, Any], request_id: str
) -> PredictionResponse:
    """Pack a ModelService.predict() output into the wire response model."""
    return PredictionResponse(
        predicted_classes=[
            PredictedClass(**{"class": p["class"], "confidence": float(p["confidence"])})
            for p in prediction["predicted_classes"]
        ],
        all_scores={k: float(v) for k, v in prediction["all_scores"].items()},
        threshold=float(prediction["threshold"]),
        processing_time_ms=int(prediction["processing_time_ms"]),
        model_version=str(prediction["model_version"]),
        request_id=request_id,
    )


_INSERT_PREDICTION_CLASS_SQL = text(
    """
    INSERT INTO prediction_classes (prediction_id, class_label, confidence)
    VALUES (:prediction_id, :class_label, :confidence)
    """
)


async def _persist_multi_label_prediction(
    db: AsyncSession,
    prediction_id: uuid.UUID,
    predicted_classes: list[dict[str, Any]],
) -> None:
    """Insert one row per detected class into ``prediction_classes``.

    Args:
        db: Open async DB session.
        prediction_id: UUID of the parent ``predictions`` row. Same as the
            request_id returned to the client.
        predicted_classes: List of ``{class, confidence}`` dicts (the
            ``predicted_classes`` field of the ModelService output).

    Returns:
        None. The function never raises — any database error is logged.
    """
    if not predicted_classes:
        # Empty list is a valid prediction; nothing to insert.
        return
    try:
        for entry in predicted_classes:
            await db.execute(
                _INSERT_PREDICTION_CLASS_SQL,
                {
                    "prediction_id": str(prediction_id),
                    "class_label": str(entry["class"]),
                    "confidence": float(entry["confidence"]),
                },
            )
        await db.commit()
    except SQLAlchemyError as exc:
        logger.warning(
            "prediction_classes persist failed for %s: %s", prediction_id, exc
        )
        try:
            await db.rollback()
        except SQLAlchemyError:
            logger.exception("rollback also failed for %s", prediction_id)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "prediction_classes payload was malformed for %s: %s",
            prediction_id,
            exc,
        )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict(
    request: Request,
    audio_file: UploadFile = File(...),
    model_service: ModelService = Depends(get_model_service),
    db: AsyncSession = Depends(get_db),
) -> PredictionResponse:
    """Run multi-label inference on a single WAV upload."""
    max_size_mb = getattr(request.app.state.settings, "MAX_AUDIO_SIZE_MB", 10)
    audio_bytes = await validate_audio_upload(audio_file, max_size_mb)
    # request_id and the parent predictions.id are the same UUID by design.
    request_uuid = uuid.uuid4()
    request_id = str(request_uuid)
    try:
        prediction = model_service.predict(audio_bytes)
    except InvalidAudioError as exc:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error_code="UNPROCESSABLE_AUDIO",
                message="Audio could not be processed",
                detail=str(exc),
            ).model_dump(),
        ) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error_code="SERVICE_UNAVAILABLE",
                message="Service is not ready",
                detail=str(exc),
            ).model_dump(),
        ) from exc
    except RequestSizeLimitExceeded:
        raise
    except PredictionError as exc:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="MODEL_ERROR",
                message="Inference failed",
                detail=str(exc),
            ).model_dump(),
        ) from exc

    response = _build_prediction_response(prediction, request_id)
    # Persist the per-class breakdown. Failures are absorbed inside the helper
    # so a flaky DB never blocks the inference response.
    await _persist_multi_label_prediction(
        db, request_uuid, prediction["predicted_classes"]
    )
    return response


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    model_service: ModelService = Depends(get_model_service),
) -> HealthResponse:
    """Liveness + readiness."""
    model_loaded = bool(model_service.is_loaded())
    version = getattr(request.app.state, "service_version", "0.0.0")
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        version=version,
        uptime_seconds=_uptime_seconds(),
    )


@router.get("/classes", response_model=ClassesResponse)
async def get_classes() -> ClassesResponse:
    """Return the 7-class taxonomy and its integer ID mapping."""
    return ClassesResponse(
        classes=list(CLASS_LABELS),
        label_to_id=dict(LABEL2ID),
        id_to_label={str(k): v for k, v in ID2LABEL.items()},
    )
