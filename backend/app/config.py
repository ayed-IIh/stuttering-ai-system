from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment (and optional `.env`)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    MODEL_PATH: str = Field(
        default="",
        description="Filesystem path to model artifact or local cache path for S3-backed models.",
    )
    MODEL_SOURCE: Literal["local", "s3"] = "local"

    MODEL_VERSION: str = Field(
        default="",
        description="S3 artifact version to download, e.g. v1.0. Required when MODEL_SOURCE='s3'.",
    )
    MODEL_CACHE_DIR: str = Field(
        default="/tmp/model_cache",
        description="Local directory where S3 artifacts are cached after download.",
    )

    DEVICE: Literal["cpu", "cuda"] = "cpu"
    MAX_AUDIO_SIZE_MB: int = 10
    # Stored as str so env / .env can use comma-separated values (pydantic-settings JSON-decodes list fields).
    allowed_origins_raw: str = Field(
        default="",
        validation_alias="ALLOWED_ORIGINS",
    )
    PRODUCTION_MODE: bool = False
    DB_URL: str = Field(
        default="postgresql://user:password@localhost:5432/stuttering_ai",
        description="Database URL (placeholder default for local dev).",
    )
    SERVICE_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    @field_validator("allowed_origins_raw", mode="before")
    @classmethod
    def coerce_allowed_origins_raw(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ",".join(str(x).strip() for x in value if str(x).strip())
        if isinstance(value, str):
            return value.strip()
        raise TypeError("ALLOWED_ORIGINS must be a list or string")

    @staticmethod
    def _split_origins(raw: str) -> list[str]:
        if not raw.strip():
            return []
        if raw.strip().startswith("["):
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("ALLOWED_ORIGINS JSON must be a list of strings")
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [part.strip() for part in raw.split(",") if part.strip()]

    @computed_field
    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        return self._split_origins(self.allowed_origins_raw)

    @field_validator("LOG_LEVEL")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        upper = value.strip().upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}")
        return upper

    @model_validator(mode="after")
    def validate_cors_wildcard(self) -> Settings:
        if self.PRODUCTION_MODE and "*" in self.ALLOWED_ORIGINS:
            raise ValueError(
                "ALLOWED_ORIGINS must not contain '*' when PRODUCTION_MODE is true"
            )
        return self

    @model_validator(mode="after")
    def validate_s3_fields(self) -> Settings:
        if self.MODEL_SOURCE == "s3" and not self.MODEL_VERSION.strip():
            raise ValueError(
                "MODEL_VERSION is required when MODEL_SOURCE='s3'. "
                "Set it in .env or as an environment variable."
            )
        return self

    @property
    def cors_allowed_origins(self) -> list[str]:
        """Origins passed to CORSMiddleware (never `['*']` in production)."""
        if self.PRODUCTION_MODE:
            return [o for o in self.ALLOWED_ORIGINS if o != "*"]
        return list(self.ALLOWED_ORIGINS)

    @property
    def max_audio_size_bytes(self) -> int:
        return self.MAX_AUDIO_SIZE_MB * 1024 * 1024

    @property
    def resolved_model_path(self) -> Path:
        """
        Returns the local path where model_inference.pt lives.

        For local source: returns MODEL_PATH directly.
        For S3 source:    returns the cache dir where download_model pulled the artifact.
        """
        if self.MODEL_SOURCE == "s3":
            return Path(self.MODEL_CACHE_DIR) / self.MODEL_VERSION
        return Path(self.MODEL_PATH)


def download_model_if_needed(settings: Settings) -> None:
    """
    Pull model artifacts from S3 into the local cache if they aren't there yet.

    This reuses the same download + hash-verification logic from
    scripts/download_model.py so there's no duplicated S3 code.
    Skips the download entirely when MODEL_SOURCE='local'.
    """
    if settings.MODEL_SOURCE != "s3":
        return

    # Import here to avoid a hard boto3 dependency when running locally
    from scripts.download_model import download_and_verify
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    cache_path = Path(settings.MODEL_CACHE_DIR) / settings.MODEL_VERSION

    # Already cached — check if both files are present before skipping
    artifacts_present = all(
        (cache_path / f).exists()
        for f in ["model_inference.pt", "config.json"]
    )
    if artifacts_present:
        logger.info(
            "Model artifacts already cached at %s — skipping S3 download.", cache_path
        )
        return

    logger.info(
        "MODEL_SOURCE=s3 — downloading version '%s' from S3 into %s",
        settings.MODEL_VERSION,
        cache_path,
    )

    try:
        s3_client = boto3.client("s3")
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to create S3 client: {exc}") from exc
    # download_and_verify handles mkdir, download, and MD5 check
    download_and_verify(s3_client, settings.MODEL_VERSION, cache_path)

    logger.info("S3 model artifacts ready at %s", cache_path)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
