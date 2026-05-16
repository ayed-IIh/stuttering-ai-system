from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Default cache lives under the calling user's home (XDG-style) rather than
# world-writable /tmp, so cached model artifacts can't be tampered with by other
# local users. Override with the MODEL_CACHE_DIR env var.
DEFAULT_MODEL_CACHE_DIR = Path.home() / ".cache" / "stuttering_ai" / "model_cache"


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
        default_factory=lambda: str(DEFAULT_MODEL_CACHE_DIR),
        description=(
            "Local directory where S3 artifacts are cached after download. "
            "Defaults to an app-owned path under the user's home; the leaf is "
            "created with mode 0o700 on first use. Override with MODEL_CACHE_DIR."
        ),
    )

    DEVICE: Literal["cpu", "cuda"] = "cpu"
    MAX_AUDIO_SIZE_MB: int = 10
    # Multi-label decision threshold: sigmoid(logit) >= threshold ⇒ class is in
    # ``predicted_classes``. Per-class threshold tuning happens at eval time; this
    # is the global default applied at /predict.
    MULTI_LABEL_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
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
        if self.MODEL_SOURCE != "s3":
            return self

        version = self.MODEL_VERSION.strip()
        if not version:
            raise ValueError(
                "MODEL_VERSION is required when MODEL_SOURCE='s3'. "
                "Set it in .env or as an environment variable."
            )

        # MODEL_VERSION is concatenated onto MODEL_CACHE_DIR to build the
        # artifact path. A value like "../etc" or "/abs/path" would escape the
        # cache root, so reject anything that isn't a plain version token.
        version_parts = PurePosixPath(version).parts
        if (
            "/" in version
            or "\\" in version
            or PurePosixPath(version).is_absolute()
            or ".." in version_parts
        ):
            raise ValueError(
                "MODEL_VERSION must be a simple version token (e.g. 'v1.0.0') — "
                "no path separators, parent references, or absolute paths."
            )

        self.MODEL_VERSION = version
        return self

    @model_validator(mode="after")
    def validate_production_model_config(self) -> Settings:
        if (
            self.PRODUCTION_MODE
            and self.MODEL_SOURCE == "local"
            and not self.MODEL_PATH.strip()
        ):
            raise ValueError(
                "MODEL_PATH is required when PRODUCTION_MODE=true and "
                "MODEL_SOURCE='local'. Set it to the directory containing "
                "model_inference.pt and config.json."
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


def _cached_artifacts_match_s3(
    s3_client,
    version: str,
    cache_path: Path,
    artifacts: list[str],
    bucket: str,
    md5_of_file,
    strip_etag_quotes,
) -> bool:
    """Return True iff every cached artifact's MD5 matches the current S3 ETag.

    Args:
        s3_client: A live boto3 S3 client.
        version: The MODEL_VERSION token (used as the S3 key prefix).
        cache_path: Local directory holding the cached files.
        artifacts: Filenames that must each be present + verified.
        bucket: S3 bucket name.
        md5_of_file: Helper computing the MD5 of a local file.
        strip_etag_quotes: Helper stripping the wrapping quotes from an S3 ETag.

    Returns:
        True if every artifact matches its expected MD5. False on any HEAD
        failure or hash mismatch (the caller should then re-download).
    """
    # Local import keeps botocore optional for non-S3 deployments.
    from botocore.exceptions import ClientError

    for filename in artifacts:
        local_path = cache_path / filename
        s3_key = f"{version}/{filename}"
        try:
            head = s3_client.head_object(Bucket=bucket, Key=s3_key)
        except ClientError as exc:
            logger.warning(
                "Cache verification HEAD failed for s3://%s/%s: %s — will re-download.",
                bucket, s3_key, exc,
            )
            return False
        expected = strip_etag_quotes(head["ETag"])
        actual = md5_of_file(local_path)
        if actual != expected:
            logger.warning(
                "Cache integrity mismatch for %s (expected %s, got %s) — will re-download.",
                filename, expected, actual,
            )
            return False
    return True


def download_model_if_needed(settings: Settings) -> None:
    """Pull model artifacts from S3 into the local cache, verifying integrity.

    Behaviour:
        * MODEL_SOURCE != 's3'  → no-op.
        * Cache directory missing or incomplete  → download + verify.
        * Cache directory present but any artifact's MD5 doesn't match the
          current S3 ETag → re-download + verify.
        * All cached artifacts match S3 → skip the download.

    The cache leaf is created with mode 0o700 on first use to keep cached
    artifacts isolated from other local users.
    """
    if settings.MODEL_SOURCE != "s3":
        return

    # Import here to avoid a hard boto3 dependency when running locally
    from scripts.download_model import (
        ARTIFACTS,
        BUCKET,
        download_and_verify,
        md5_of_file,
        strip_etag_quotes,
    )
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    cache_path = Path(settings.MODEL_CACHE_DIR) / settings.MODEL_VERSION
    # Restrictive perms; on Windows mode is effectively no-op but still safe.
    cache_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        cache_path.chmod(0o700)
    except OSError as exc:
        logger.warning("Could not enforce 0o700 on %s: %s", cache_path, exc)

    try:
        s3_client = boto3.client("s3")
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to create S3 client: {exc}") from exc

    artifacts_present = all((cache_path / f).exists() for f in ARTIFACTS)
    if artifacts_present and _cached_artifacts_match_s3(
        s3_client,
        settings.MODEL_VERSION,
        cache_path,
        ARTIFACTS,
        BUCKET,
        md5_of_file,
        strip_etag_quotes,
    ):
        logger.info(
            "Cached model artifacts at %s passed MD5 integrity check — skipping S3 download.",
            cache_path,
        )
        return

    logger.info(
        "MODEL_SOURCE=s3 — downloading version '%s' from S3 into %s",
        settings.MODEL_VERSION,
        cache_path,
    )
    # download_and_verify handles mkdir, download, and MD5 check
    download_and_verify(s3_client, settings.MODEL_VERSION, cache_path)

    logger.info("S3 model artifacts ready at %s", cache_path)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
