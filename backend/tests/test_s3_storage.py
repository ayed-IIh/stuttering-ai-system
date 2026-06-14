from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from backend.services.s3_storage import S3Error, S3Storage

BUCKET = "stt-test-bucket"


@pytest.fixture
def storage():
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield S3Storage(BUCKET, region="us-east-1")


def test_upload_download_roundtrip(storage: S3Storage) -> None:
    storage.upload_bytes("feedback/x.wav", b"RIFFdata", content_type="audio/wav")
    assert storage.download_bytes("feedback/x.wav") == b"RIFFdata"


def test_list_keys_filters_by_prefix(storage: S3Storage) -> None:
    storage.upload_bytes("feedback/a.json", b"{}")
    storage.upload_bytes("feedback/b.json", b"{}")
    storage.upload_bytes("predict-audio/c.wav", b"x")
    assert sorted(storage.list_keys("feedback/")) == [
        "feedback/a.json",
        "feedback/b.json",
    ]


def test_get_json(storage: S3Storage) -> None:
    storage.upload_bytes("feedback/m.json", b'{"correct_labels": ["blocks"]}')
    assert storage.get_json("feedback/m.json") == {"correct_labels": ["blocks"]}


def test_missing_key_raises_s3error(storage: S3Storage) -> None:
    with pytest.raises(S3Error):
        storage.download_bytes("feedback/does-not-exist.wav")


def test_empty_bucket_name_rejected() -> None:
    with pytest.raises(ValueError):
        S3Storage("")
