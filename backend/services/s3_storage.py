"""S3-backed storage for prediction audio and the HITL feedback corpus.

Object layout in the bucket:

  ``<predict_prefix><key>``        — a clip the mobile uploaded; the AI
                                     downloads it to run inference.
  ``<feedback_prefix><id>.wav``    — a therapist-corrected clip (retraining
                                     corpus).
  ``<feedback_prefix><id>.json``   — its metadata (correct_labels,
                                     original_prediction, model_version, ...).

S3 has no append operation, so each correction is stored as a self-contained
pair of objects; the retraining job lists ``<feedback_prefix>*.json`` to
reassemble the corpus. This keeps the store horizontally scalable (no shared
mutable manifest, unlike the single-host local file backend).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class S3Error(Exception):
    """Raised when an S3 operation fails (network, auth, missing key, ...)."""


class S3Storage:
    """Thin, dependency-light wrapper over the boto3 S3 client.

    All failures are normalized to :class:`S3Error` so callers don't have to
    know about botocore exception types.
    """

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket name is required")
        self.bucket = bucket
        # endpoint_url lets this point at an S3-compatible store or a local
        # mock; region/None defers to the standard AWS credential/region chain.
        self._client = boto3.client(
            "s3",
            region_name=region or None,
            endpoint_url=endpoint_url or None,
        )

    def download_bytes(self, key: str) -> bytes:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except (BotoCoreError, ClientError) as exc:
            raise S3Error(
                f"failed to download s3://{self.bucket}/{key}: {exc}"
            ) from exc

    def upload_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        except (BotoCoreError, ClientError) as exc:
            raise S3Error(
                f"failed to upload s3://{self.bucket}/{key}: {exc}"
            ) from exc

    def list_keys(self, prefix: str) -> list[str]:
        """List every object key under ``prefix`` (paginated)."""
        keys: list[str] = []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        except (BotoCoreError, ClientError) as exc:
            raise S3Error(
                f"failed to list s3://{self.bucket}/{prefix}: {exc}"
            ) from exc
        return keys

    def get_json(self, key: str) -> dict:
        return json.loads(self.download_bytes(key).decode("utf-8"))
