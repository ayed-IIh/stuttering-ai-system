from __future__ import annotations

import io
import logging
import time
import uuid
import wave

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.middleware import (
    RequestSizeLimitExceeded,
    validate_audio_upload,
)
from backend.db import crud
from backend.db.database import get_db
from backend.db.schemas import ConfidenceScores, PredictionCreate
from backend.services.model_service import (
    InvalidAudioError,
    ModelNotLoadedError,
    ModelService,
    PredictionError,
)
from shared.labels import CLASS_LABELS, ID2LABEL, LABEL2ID

logger = logging.getLogger(__name__)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence_scores: dict[str, float]
    processing_time_ms: int
    model_version: str
    request_id: str


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
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate <= 0:
                return 0.0
            return frames / float(rate)
    except Exception:
        return 0.0


def _confidence_scores_for_db(raw: dict[str, float]) -> ConfidenceScores:
    keys = (
        "fluent",
        "blocks",
        "interjections",
        "prolongations",
        "part_word_repetition",
        "phrase_repetition",
        "word_repetition",
    )
    norm = {str(k).lower(): float(v) for k, v in raw.items()}
    merged = {k: float(norm.get(k, 0.0)) for k in keys}
    return ConfidenceScores(**merged)


async def _persist_prediction(
    db: AsyncSession,
    *,
    request: Request,
    audio_bytes: bytes,
    audio_filename: str | None,
    response: PredictionResponse,
) -> None:
    model_version_id = await crud.get_model_version_id_by_label(
        db, response.model_version
    )
    if model_version_id is None:
        logger.debug(
            "Skipping prediction DB row: no model_versions row for %r",
            response.model_version,
        )
        return
    client = request.client
    host = client.host if client else "127.0.0.1"
    try:
        payload = PredictionCreate(
            audio_filename=(audio_filename or "upload.wav")[:500],
            audio_duration_sec=_wav_duration_sec(audio_bytes),
            predicted_class=response.predicted_class,
            confidence_scores=_confidence_scores_for_db(response.confidence_scores),
            model_version_id=model_version_id,
            processing_time_ms=response.processing_time_ms,
            client_ip=host,
            request_id=uuid.UUID(response.request_id),
        )
        await crud.create_prediction(db, payload)
    except HTTPException as exc:
        logger.warning(
            "Prediction DB insert rejected: %s — %s",
            exc.status_code,
            exc.detail,
        )
    except Exception:
        logger.exception("Prediction DB insert failed")


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
    started_at = time.perf_counter()
    max_size_mb = getattr(request.app.state.settings, "MAX_AUDIO_SIZE_MB", 10)
    audio_bytes = await validate_audio_upload(audio_file, max_size_mb)
    request_id = str(uuid.uuid4())
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

    response = PredictionResponse(
        predicted_class=str(prediction["predicted_class"]),
        confidence_scores={k: float(v) for k, v in prediction["confidence_scores"].items()},
        processing_time_ms=int((time.perf_counter() - started_at) * 1000),
        model_version=str(prediction["model_version"]),
        request_id=request_id,
    )
    await _persist_prediction(
        db,
        request=request,
        audio_bytes=audio_bytes,
        audio_filename=audio_file.filename,
        response=response,
    )
    return response


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    model_service: ModelService = Depends(get_model_service),
) -> HealthResponse:
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
    return ClassesResponse(
        classes=list(CLASS_LABELS),
        label_to_id=dict(LABEL2ID),
        id_to_label={str(k): v for k, v in ID2LABEL.items()},
    )
