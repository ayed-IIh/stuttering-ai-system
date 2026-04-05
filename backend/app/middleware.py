from __future__ import annotations

import time
from datetime import datetime, timezone

import structlog
from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)

# WAV RIFF container: bytes 0–3 are ASCII RIFF (0x52, 0x49, 0x46, 0x46)
WAV_RIFF_MAGIC = b"RIFF"

ALLOWED_AUDIO_MIME_TYPES = frozenset({"audio/wav", "audio/x-wav"})


class AudioUploadValidationError(Exception):
    status_code = 400
    error_code = "INVALID_REQUEST"
    message = "Invalid audio upload"

    def __init__(self, detail: str = "") -> None:
        super().__init__(detail)
        self.detail = detail


class InvalidAudioMimeTypeError(AudioUploadValidationError):
    status_code = 415
    error_code = "UNSUPPORTED_MEDIA_TYPE"
    message = "Only WAV files are supported"


class InvalidAudioMagicBytesError(AudioUploadValidationError):
    status_code = 400
    error_code = "INVALID_REQUEST"
    message = "Invalid WAV file signature"


class RequestSizeLimitExceeded(AudioUploadValidationError):
    status_code = 413
    error_code = "FILE_TOO_LARGE"
    message = "Audio file exceeds maximum allowed size"


async def validate_audio_upload(file: UploadFile, max_size_mb: int) -> bytes:
    """Validate uploaded WAV: MIME whitelist, optional Content-Length, RIFF magic, size cap."""
    max_size_bytes = max_size_mb * 1024 * 1024
    content_type = (file.content_type or "").lower().strip()
    if content_type not in ALLOWED_AUDIO_MIME_TYPES:
        raise InvalidAudioMimeTypeError(
            f"content_type={content_type!r}, allowed={sorted(ALLOWED_AUDIO_MIME_TYPES)}"
        )

    declared_length_header = file.headers.get("content-length")
    if declared_length_header:
        try:
            declared_length = int(declared_length_header)
        except ValueError:
            declared_length = -1
        if declared_length > max_size_bytes:
            raise RequestSizeLimitExceeded(f"Declared size exceeds max: {max_size_bytes} bytes")

    head = await file.read(12)
    await file.seek(0)
    if len(head) < 4 or head[:4] != WAV_RIFF_MAGIC:
        raise InvalidAudioMagicBytesError("Expected RIFF (52 49 46 46) at bytes 0–3")
    if len(head) < 12:
        raise InvalidAudioMagicBytesError("WAV header too short")

    payload = await file.read()
    if len(payload) > max_size_bytes:
        raise RequestSizeLimitExceeded(f"Payload exceeds max: {max_size_bytes} bytes")
    return payload


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs timestamp, method, path, status_code, processing_time_ms, x-request-id."""

    async def dispatch(self, request: Request, call_next):
        started_at = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        request_id = request.headers.get("x-request-id", "")
        logger.info(
            "request_processed",
            timestamp=datetime.now(timezone.utc).isoformat(),
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            processing_time_ms=elapsed_ms,
            request_id=request_id,
        )
        return response


def _is_production(request: Request) -> bool:
    settings = getattr(request.app.state, "settings", None)
    return bool(getattr(settings, "PRODUCTION_MODE", False))


def build_error_payload(
    request: Request,
    *,
    error_code: str,
    message: str,
    detail: str = "",
) -> dict[str, str]:
    """JSON error body. Production: only ``error_code`` and ``message`` (no ``detail``)."""
    if _is_production(request):
        return {"error_code": error_code, "message": message}
    body: dict[str, str] = {
        "error_code": error_code,
        "message": message,
        "detail": detail,
    }
    return body


def register_exception_handlers(app) -> None:
    """Register upload / HTTP error handlers (413 for ``RequestSizeLimitExceeded`` first)."""

    @app.exception_handler(RequestSizeLimitExceeded)
    async def request_size_handler(request: Request, exc: RequestSizeLimitExceeded):
        return JSONResponse(
            status_code=413,
            content=build_error_payload(
                request,
                error_code=exc.error_code,
                message=exc.message,
                detail=exc.detail,
            ),
        )

    @app.exception_handler(AudioUploadValidationError)
    async def audio_validation_handler(request: Request, exc: AudioUploadValidationError):
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(
                request,
                error_code=exc.error_code,
                message=exc.message,
                detail=exc.detail,
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            error_code = str(detail.get("error_code", "INTERNAL_ERROR"))
            message = str(detail.get("message", "Request failed"))
            detail_text = str(detail.get("detail", ""))
        else:
            error_code = "INTERNAL_ERROR"
            if _is_production(request):
                message = "Request failed"
                detail_text = ""
            else:
                message = str(detail) if detail else "Request failed"
                detail_text = str(detail) if detail else ""
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(
                request,
                error_code=error_code,
                message=message,
                detail=detail_text,
            ),
        )
