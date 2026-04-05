"""Tests for backend.app.middleware (upload validation, error payloads, logging)."""

from __future__ import annotations

import asyncio
import io
import wave
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import Headers, UploadFile

from backend.app.middleware import (
    InvalidAudioMagicBytesError,
    InvalidAudioMimeTypeError,
    RequestLoggingMiddleware,
    RequestSizeLimitExceeded,
    register_exception_handlers,
    validate_audio_upload,
)


def _upload(
    content: bytes,
    content_type: str,
    *,
    content_length: str | None = None,
    filename: str = "clip.wav",
) -> UploadFile:
    hdr: dict[str, str] = {"content-type": content_type}
    if content_length is not None:
        hdr["content-length"] = content_length
    headers = Headers(hdr)
    return UploadFile(file=io.BytesIO(content), headers=headers, filename=filename)


def _minimal_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8_000)
        w.writeframes(b"\x00\x01" * 400)
    data = buf.getvalue()
    assert data[:4] == b"RIFF"
    return data


def test_validate_rejects_disallowed_mime():
    async def run():
        uf = _upload(b"RIFFxxxxWAVEfmt data", "application/octet-stream")
        with pytest.raises(InvalidAudioMimeTypeError):
            await validate_audio_upload(uf, max_size_mb=10)

    asyncio.run(run())


def test_validate_rejects_audio_wave_alias_when_not_whitelisted():
    async def run():
        uf = _upload(_minimal_wav_bytes(), "audio/wave")
        with pytest.raises(InvalidAudioMimeTypeError):
            await validate_audio_upload(uf, max_size_mb=10)

    asyncio.run(run())


def test_validate_rejects_declared_content_length_over_max():
    async def run():
        body = _minimal_wav_bytes()
        uf = _upload(body, "audio/wav", content_length=str(50 * 1024 * 1024))
        with pytest.raises(RequestSizeLimitExceeded):
            await validate_audio_upload(uf, max_size_mb=1)

    asyncio.run(run())


def test_validate_rejects_missing_riff_magic():
    async def run():
        uf = _upload(b"XXXXnotawav", "audio/wav")
        with pytest.raises(InvalidAudioMagicBytesError):
            await validate_audio_upload(uf, max_size_mb=10)

    asyncio.run(run())


def test_validate_rejects_short_header():
    async def run():
        uf = _upload(b"RIFF", "audio/wav")
        with pytest.raises(InvalidAudioMagicBytesError):
            await validate_audio_upload(uf, max_size_mb=10)

    asyncio.run(run())


def test_validate_rejects_payload_over_max():
    async def run():
        body = _minimal_wav_bytes()
        uf = _upload(body, "audio/wav")
        with pytest.raises(RequestSizeLimitExceeded):
            await validate_audio_upload(uf, max_size_mb=0)

    asyncio.run(run())


def test_validate_accepts_valid_wav():
    async def run():
        body = _minimal_wav_bytes()
        uf = _upload(body, "audio/x-wav")
        out = await validate_audio_upload(uf, max_size_mb=10)
        assert out == body

    asyncio.run(run())


def test_validate_accepts_audio_wav_mime():
    async def run():
        body = _minimal_wav_bytes()
        uf = _upload(body, "audio/wav")
        out = await validate_audio_upload(uf, max_size_mb=10)
        assert out == body

    asyncio.run(run())


def _make_app(*, production: bool) -> FastAPI:
    app = FastAPI()

    class _Cfg:
        PRODUCTION_MODE = production

    app.state.settings = _Cfg()
    register_exception_handlers(app)

    @app.post("/oversize")
    async def oversize():
        raise RequestSizeLimitExceeded("internal byte count 99999")

    @app.get("/http_raw")
    async def http_raw():
        raise HTTPException(status_code=400, detail="/secret/path failed")

    return app


def test_request_size_limit_returns_413_and_schema():
    client = TestClient(_make_app(production=False))
    r = client.post("/oversize")
    assert r.status_code == 413
    j = r.json()
    assert j["error_code"] == "FILE_TOO_LARGE"
    assert "detail" in j


def test_production_error_has_only_error_code_and_message():
    client = TestClient(_make_app(production=True))
    r = client.post("/oversize")
    assert r.status_code == 413
    j = r.json()
    assert set(j.keys()) == {"error_code", "message"}
    assert j["error_code"] == "FILE_TOO_LARGE"


def test_production_http_exception_string_detail_is_generic():
    client = TestClient(_make_app(production=True))
    r = client.get("/http_raw")
    assert r.status_code == 400
    j = r.json()
    assert set(j.keys()) == {"error_code", "message"}
    assert "secret" not in j["message"].lower()


def test_dev_http_exception_includes_detail():
    client = TestClient(_make_app(production=False))
    r = client.get("/http_raw")
    j = r.json()
    assert "detail" in j
    assert "secret" in j["detail"]


@patch("backend.app.middleware.logger")
def test_request_logging_middleware_logs_fields(mock_logger):
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping")
    async def ping():
        return {"pong": True}

    client = TestClient(app)
    client.get("/ping", headers={"x-request-id": "test-req-id"})

    mock_logger.info.assert_called_once()
    call = mock_logger.info.call_args
    assert call.args[0] == "request_processed"
    kwargs = call.kwargs
    assert kwargs.get("method") == "GET"
    assert kwargs.get("path") == "/ping"
    assert kwargs.get("status_code") == 200
    assert kwargs.get("request_id") == "test-req-id"
    assert "processing_time_ms" in kwargs
    assert "timestamp" in kwargs
