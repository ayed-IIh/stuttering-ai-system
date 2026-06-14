from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

# Repo root is on sys.path via backend/tests/conftest.py, so scripts.* imports.
from backend.services.s3_storage import S3Storage
from scripts.retrain import (
    Correction,
    build_augmented_manifest,
    load_corpus_local,
    load_corpus_s3,
    stage_corpus,
)

BUCKET = "retrain-test"


def test_load_corpus_local(tmp_path: Path) -> None:
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "a.wav").write_bytes(b"RIFFdata")
    rec = {"id": "a", "audio_file": "audio/a.wav", "correct_labels": ["blocks"]}
    (tmp_path / "feedback.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

    corpus = load_corpus_local(str(tmp_path))
    assert len(corpus) == 1
    assert corpus[0].label == "blocks"
    assert corpus[0].audio_bytes == b"RIFFdata"


def test_load_corpus_local_skips_unknown_label(tmp_path: Path) -> None:
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "a.wav").write_bytes(b"x")
    rec = {"id": "a", "audio_file": "audio/a.wav", "correct_labels": ["NOT_A_CLASS"]}
    (tmp_path / "feedback.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

    assert load_corpus_local(str(tmp_path)) == []


def test_load_corpus_s3() -> None:
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        s3 = S3Storage(BUCKET, region="us-east-1")
        s3.upload_bytes("feedback/a.wav", b"RIFFdata", "audio/wav")
        s3.upload_bytes(
            "feedback/a.json",
            json.dumps(
                {"id": "a", "audio_file": "feedback/a.wav", "correct_labels": ["fluent"]}
            ).encode(),
        )
        corpus = load_corpus_s3(s3, "feedback/")
        assert len(corpus) == 1
        assert corpus[0].label == "fluent"


def test_build_augmented_manifest(tmp_path: Path) -> None:
    base = tmp_path / "base.csv"
    base.write_text("audio_path,label\n/data/x.wav,fluent\n", encoding="utf-8")
    rows = [("/staged/a.wav", "blocks"), ("/staged/b.wav", "word_repetition")]
    out = tmp_path / "aug.csv"

    total = build_augmented_manifest(base, rows, out)
    assert total == 3
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "audio_path,label"
    assert "/staged/a.wav,blocks" in text
    assert "/staged/b.wav,word_repetition" in text


def test_build_augmented_manifest_requires_columns(tmp_path: Path) -> None:
    base = tmp_path / "bad.csv"
    base.write_text("path,klass\n/x.wav,fluent\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_augmented_manifest(base, [], tmp_path / "out.csv")


def test_stage_corpus_writes_audio(tmp_path: Path) -> None:
    rows = stage_corpus(
        [Correction(id="x", label="blocks", audio_bytes=b"RIFFxx")], tmp_path / "staged"
    )
    assert len(rows) == 1
    path, label = rows[0]
    assert label == "blocks"
    assert Path(path).read_bytes() == b"RIFFxx"
