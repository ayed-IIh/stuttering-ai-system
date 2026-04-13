"""
model_loader.py — loads model_inference.pt from either local disk or S3.

The caller decides the source by setting MODEL_SOURCE in the environment:
    MODEL_SOURCE=local   →  reads MODEL_LOCAL_PATH (default: models/model_inference.pt)
    MODEL_SOURCE=s3      →  downloads from S3 using MODEL_VERSION

Required env vars when MODEL_SOURCE=s3:
    MODEL_VERSION          e.g. v1.0
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION

Optional:
    MODEL_CACHE_DIR        where to cache the S3 download (default: /tmp/model_cache)
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import boto3
import torch
from botocore.exceptions import BotoCoreError, ClientError

load_dotenv()

logger = logging.getLogger(__name__)

BUCKET = "stuttering-ai-models"
ARTIFACTS = ["model_inference.pt", "config.json"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _s3_etag(s3_client, key: str) -> str:
    head = s3_client.head_object(Bucket=BUCKET, Key=key)
    return head["ETag"].strip('"')


def _download_from_s3(version: str, dest_dir: Path) -> None:
    """Pull artifacts for a version into dest_dir, verifying MD5 after each download."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    for filename in ARTIFACTS:
        s3_key = f"{version}/{filename}"
        local_path = dest_dir / filename

        # Skip re-download if the cached file's hash still matches S3
        if local_path.exists():
            expected = _s3_etag(s3, s3_key)
            if _md5(local_path) == expected:
                logger.info("Cache hit for %s — skipping download.", filename)
                continue

        logger.info("Downloading s3://%s/%s → %s", BUCKET, s3_key, local_path)
        s3.download_file(Bucket=BUCKET, Key=s3_key, Filename=str(local_path))

        actual = _md5(local_path)
        expected = _s3_etag(s3, s3_key)
        if actual != expected:
            local_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Hash mismatch for {filename} after download "
                f"(expected {expected}, got {actual}). File removed."
            )
        logger.info("Verified %s  MD5=%s…", filename, actual[:12])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model_and_config() -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Return (state_dict, config) ready for your model class to consume.

    This is the single entry point the app should call — it handles both
    local and S3 sources transparently.
    """
    source = os.environ.get("MODEL_SOURCE", "local").lower()

    if source == "s3":
        version = os.environ.get("MODEL_VERSION")
        if not version:
            raise EnvironmentError("MODEL_VERSION must be set when MODEL_SOURCE=s3")

        cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/model_cache")) / version

        try:
            _download_from_s3(version, cache_dir)
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(f"Failed to download model from S3: {exc}") from exc

        model_path = cache_dir / "model_inference.pt"
        config_path = cache_dir / "config.json"

    elif source == "local":
        local_path = Path(os.environ.get("MODEL_LOCAL_PATH", "models/model_inference.pt"))
        model_path = local_path
        config_path = local_path.parent / "config.json"

    else:
        raise ValueError(f"Unknown MODEL_SOURCE '{source}'. Expected 'local' or 's3'.")

    if not model_path.exists():
        raise FileNotFoundError(f"model_inference.pt not found at {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")

    # map_location=cpu so the loader works even on machines without a GPU
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    with open(config_path) as fh:
        config = json.load(fh)

    logger.info("Model loaded from '%s' source (path: %s)", source, model_path)
    return state_dict, config