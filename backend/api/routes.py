from __future__ import annotations

import io
import logging
import time
import uuid
import wave
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.middleware import (
    RequestSizeLimitExceeded,
    validate_audio_upload,
)
from backend.db import crud
from backend.db.database import get_db
from backend.db.schemas import ConfidenceScores, PredictionCreate
from backend.services.feedback_service import FeedbackError, FeedbackStore
from backend.services.model_service import (
    InvalidAudioError,
    ModelNotLoadedError,
    ModelService,
    PredictionError,
)
from shared.labels import CLASS_LABELS, ID2LABEL, LABEL2ID

logger = logging.getLogger(__name__)


class PredictedClass(BaseModel):
    """One entry in the ``predicted_classes`` array of a /predict response.

    The mobile client (Laravel ``AiProcessingService``) reads each entry's
    ``class`` and ``confidence`` keys. ``class`` is a Python keyword, so it is
    exposed via an alias and serialized as ``class`` in the JSON body.
    """

    model_config = ConfigDict(populate_by_name=True)

    label: str = Field(serialization_alias="class", validation_alias="class")
    confidence: float


class PredictionResponse(BaseModel):
    # --- Native single-label fields (kept for the DB layer and direct consumers) ---
    predicted_class: str
    confidence_scores: dict[str, float]
    processing_time_ms: int
    model_version: str
    request_id: str
    # --- Multi-label-shaped fields the mobile contract requires ---
    # The mobile throws if any of these is missing/wrong-typed. For single-label
    # output, ``predicted_classes`` holds exactly one entry (the argmax winner),
    # ``all_scores`` is the full softmax map (0-1), and ``threshold`` equals the
    # winner's confidence so history-recompute (all_scores >= threshold) also
    # yields just the winner.
    predicted_classes: list[PredictedClass] = []
    all_scores: dict[str, float] = {}
    threshold: float = 0.0


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


class FeedbackRequest(BaseModel):
    """HITL correction posted by the mobile client when a therapist edits the
    AI diagnosis. Matches the Laravel ``DiagnosisConfirmationService`` payload.

    ``correct_labels`` / ``original_prediction`` are accepted loosely (strings
    or ``{class}`` objects) and normalized by the feedback store.
    """

    audio_base64: str
    correct_labels: str | dict[str, Any] | list[Any] | None = None
    original_prediction: str | dict[str, Any] | list[Any] | None = None
    model_version: str = "unknown"


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str | None = None
    stored_count: int | None = None


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


def get_feedback_store(request: Request) -> FeedbackStore:
    store = getattr(request.app.state, "feedback_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error_code="SERVICE_UNAVAILABLE",
                message="Service is not ready",
                detail="Feedback store is not initialized",
            ).model_dump(),
        )
    return store


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


def _multilabel_shim(
    predicted_class: str,
    confidence_scores: dict[str, float],
) -> tuple[list[PredictedClass], dict[str, float], float]:
    """Map single-label output onto the multi-label-shaped contract the mobile
    client requires.

    Returns ``(predicted_classes, all_scores, threshold)`` where
    ``predicted_classes`` is a one-element list (the argmax winner),
    ``all_scores`` is the full 0-1 softmax map, and ``threshold`` equals the
    winner's confidence (so any consumer that recomputes the prediction as
    ``all_scores >= threshold`` still selects exactly the winner).
    """
    all_scores = {str(k): float(v) for k, v in confidence_scores.items()}
    winner_conf = all_scores.get(predicted_class)
    if winner_conf is None:
        winner_conf = max(all_scores.values(), default=0.0)
    predicted_classes = [
        PredictedClass(label=str(predicted_class), confidence=float(winner_conf))
    ]
    return predicted_classes, all_scores, float(winner_conf)


async def _persist_prediction(
    db: AsyncSession | None,
    *,
    request: Request,
    audio_bytes: bytes,
    audio_filename: str | None,
    response: PredictionResponse,
) -> None:
    if db is None:
        # Database disabled (no POSTGRES_* configured) — skip logging silently.
        return
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
    db: AsyncSession | None = Depends(get_db),
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

    predicted_class = str(prediction["predicted_class"])
    confidence_scores = {
        k: float(v) for k, v in prediction["confidence_scores"].items()
    }
    predicted_classes, all_scores, threshold = _multilabel_shim(
        predicted_class, confidence_scores
    )
    response = PredictionResponse(
        predicted_class=predicted_class,
        confidence_scores=confidence_scores,
        processing_time_ms=int((time.perf_counter() - started_at) * 1000),
        model_version=str(prediction["model_version"]),
        request_id=request_id,
        predicted_classes=predicted_classes,
        all_scores=all_scores,
        threshold=threshold,
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


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def feedback(
    payload: FeedbackRequest,
    store: FeedbackStore = Depends(get_feedback_store),
) -> FeedbackResponse:
    """Store a therapist correction (HITL) for later retraining.

    The mobile client fires this fire-and-forget when a diagnosis is edited; it
    expects a fast ``{"status": "accepted"}``. Declared ``def`` (not ``async``)
    on purpose: the base64 decode + WAV parse + disk writes are blocking, so
    FastAPI runs this in its threadpool instead of stalling the event loop.
    """
    try:
        record = store.save(
            audio_base64=payload.audio_base64,
            correct_labels=payload.correct_labels,
            original_prediction=payload.original_prediction,
            model_version=payload.model_version,
        )
    except FeedbackError as exc:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error_code="INVALID_FEEDBACK",
                message="Feedback could not be stored",
                detail=str(exc),
            ).model_dump(),
        ) from exc
    return FeedbackResponse(
        status="accepted",
        feedback_id=str(record["id"]),
        stored_count=store.count(),
    )
