"""
tests/test_model_pipeline.py

Unit tests for upload_model.py helpers, download_model.py helpers, and
backend/app/model_loader.py.  All S3 calls are mocked — no real AWS
credentials are needed to run these.

Run with:
    pytest tests/test_model_pipeline.py -v
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Make the repo root importable regardless of where pytest is invoked from
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.download_model import (
    BUCKET as DOWNLOAD_BUCKET,
    download_and_verify,
    md5_of_file,
    strip_etag_quotes,
)
from scripts.upload_model import (
    build_tags,
    sha256_of_file,
    validate_source_dir,
)


# ---------------------------------------------------------------------------
# upload_model helpers
# ---------------------------------------------------------------------------

class TestSha256OfFile:
    def test_known_content(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert sha256_of_file(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert sha256_of_file(f) == expected


class TestBuildTags:
    def test_tag_keys_present(self):
        tags = build_tags("my_experiment", 0.912)
        keys = {t["Key"] for t in tags}
        assert keys == {"experiment_name", "val_f1", "upload_date"}

    def test_val_f1_formatted(self):
        tags = build_tags("run1", 0.8)
        val_tag = next(t for t in tags if t["Key"] == "val_f1")
        # Must be a decimal string, not scientific notation
        assert "." in val_tag["Value"]
        assert float(val_tag["Value"]) == pytest.approx(0.8)

    def test_experiment_name_preserved(self):
        tags = build_tags("stuttering_wav2vec_run3", 0.5)
        exp_tag = next(t for t in tags if t["Key"] == "experiment_name")
        assert exp_tag["Value"] == "stuttering_wav2vec_run3"


class TestValidateSourceDir:
    def test_passes_when_all_files_present(self, tmp_path):
        (tmp_path / "model_inference.pt").write_bytes(b"fake")
        (tmp_path / "config.json").write_text("{}")
        validate_source_dir(tmp_path)  # should not raise

    def test_exits_when_file_missing(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        # model_inference.pt is absent — expect sys.exit(1)
        with pytest.raises(SystemExit) as exc_info:
            validate_source_dir(tmp_path)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# download_model helpers
# ---------------------------------------------------------------------------

class TestStripEtagQuotes:
    def test_strips_double_quotes(self):
        assert strip_etag_quotes('"abc123"') == "abc123"

    def test_no_quotes_unchanged(self):
        assert strip_etag_quotes("abc123") == "abc123"


class TestMd5OfFile:
    def test_known_content(self, tmp_path):
        f = tmp_path / "f.bin"
        f.write_bytes(b"test data")
        expected = hashlib.md5(b"test data").hexdigest()
        assert md5_of_file(f) == expected


class TestDownloadAndVerify:
    """Mock S3 so we never hit the network."""

    def _write_artifacts(self, directory: Path) -> dict[str, str]:
        """Write fake artifacts and return their MD5s."""
        files = {
            "model_inference.pt": b"fake model weights",
            "config.json": b'{"label2id": {}}',
        }
        md5s = {}
        for name, content in files.items():
            (directory / name).write_bytes(content)
            md5s[name] = hashlib.md5(content).hexdigest()
        return md5s

    def test_successful_download_and_verify(self, tmp_path):
        # Pre-populate a 'remote' dir to simulate S3 content
        remote = tmp_path / "remote"
        remote.mkdir()
        md5s = self._write_artifacts(remote)

        local_out = tmp_path / "local"

        s3_mock = MagicMock()

        def fake_head(Bucket, Key):
            filename = Key.split("/")[-1]
            return {"ETag": f'"{md5s[filename]}"'}

        def fake_download(Bucket, Key, Filename):
            filename = Key.split("/")[-1]
            Path(Filename).write_bytes((remote / filename).read_bytes())

        s3_mock.head_object.side_effect = fake_head
        s3_mock.download_file.side_effect = fake_download

        download_and_verify(s3_mock, "v1.0", local_out)

        assert (local_out / "model_inference.pt").exists()
        assert (local_out / "config.json").exists()

    def test_hash_mismatch_causes_exit(self, tmp_path):
        local_out = tmp_path / "local"
        s3_mock = MagicMock()

        # head_object reports a hash that will never match what we write
        s3_mock.head_object.return_value = {"ETag": '"000000000000000000000000deadbeef"'}

        def fake_download(Bucket, Key, Filename):
            Path(Filename).write_bytes(b"corrupted data")

        s3_mock.download_file.side_effect = fake_download

        with pytest.raises(SystemExit) as exc_info:
            download_and_verify(s3_mock, "v1.0", local_out)

        assert exc_info.value.code == 1
        # The corrupt file must be cleaned up
        assert not (local_out / "model_inference.pt").exists()


# ---------------------------------------------------------------------------
# model_loader (backend)
# ---------------------------------------------------------------------------

class TestModelLoader:
    def test_load_local_source(self, tmp_path, monkeypatch):
        import torch

        state = {"fc.weight": torch.zeros(2, 3)}
        config = {"num_labels": 2, "label2id": {"fluent": 0, "stutter": 1}}

        model_pt = tmp_path / "model_inference.pt"
        config_json = tmp_path / "config.json"
        torch.save(state, model_pt)
        config_json.write_text(json.dumps(config))

        monkeypatch.setenv("MODEL_SOURCE", "local")
        monkeypatch.setenv("MODEL_LOCAL_PATH", str(model_pt))

        from backend.app.model_loader import load_model_and_config

        loaded_state, loaded_config = load_model_and_config()
        assert "fc.weight" in loaded_state
        assert loaded_config["num_labels"] == 2

    def test_unknown_source_raises(self, monkeypatch):
        monkeypatch.setenv("MODEL_SOURCE", "gcs")

        from backend.app.model_loader import load_model_and_config

        with pytest.raises(ValueError, match="Unknown MODEL_SOURCE"):
            load_model_and_config()

    def test_s3_source_missing_version_raises(self, monkeypatch):
        monkeypatch.setenv("MODEL_SOURCE", "s3")
        monkeypatch.delenv("MODEL_VERSION", raising=False)

        from backend.app.model_loader import load_model_and_config

        with pytest.raises(EnvironmentError, match="MODEL_VERSION"):
            load_model_and_config()

    @patch("backend.app.model_loader._download_from_s3")
    def test_s3_source_calls_download(self, mock_download, tmp_path, monkeypatch):
        import torch

        # Fake the cache directory that _download_from_s3 would populate
        cache = tmp_path / "v2.0"
        cache.mkdir()
        state = {"layer.bias": torch.ones(4)}
        config = {"num_labels": 4}
        torch.save(state, cache / "model_inference.pt")
        (cache / "config.json").write_text(json.dumps(config))

        monkeypatch.setenv("MODEL_SOURCE", "s3")
        monkeypatch.setenv("MODEL_VERSION", "v2.0")
        monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path))

        from backend.app.model_loader import load_model_and_config

        loaded_state, loaded_config = load_model_and_config()
        mock_download.assert_called_once()
        assert "layer.bias" in loaded_state