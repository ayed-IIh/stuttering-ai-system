"""Tests for ai.dataset.stuttering_dataset (multi-label)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from ai.dataset.stuttering_dataset import (
    StutteringDataset,
    _parse_label_cell,
    get_dataloader,
)
from shared.labels import CLASS_LABELS, LABEL2ID, NUM_CLASSES

_TARGET_SR = 16_000
_TARGET_SAMPLES = 160_000


@pytest.fixture
def fake_loader():
    """Audio loader that never touches disk — returns a fixed zero waveform."""

    def _loader(file_path: str, target_sr: int, training: bool) -> tuple[torch.Tensor, int]:
        _ = (file_path, training)
        return torch.zeros((1, _TARGET_SAMPLES), dtype=torch.float32), int(target_sr)

    return _loader


@pytest.fixture
def default_config() -> dict:
    return {
        "target_sr": _TARGET_SR,
        "max_duration_sec": 10.0,
        "normalize_method": "peak",
        "trim_silence": False,
    }


def _write_manifest(tmp_path: Path, rows: list[dict], name: str = "manifest.csv") -> str:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


class TestParseLabelCell:
    """``_parse_label_cell`` is the low-level normalizer."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("fluent", ["fluent"]),
            ("blocks,prolongations", ["blocks", "prolongations"]),
            (" blocks , prolongations ", ["blocks", "prolongations"]),
            ("a,,b", ["a", "b"]),
        ],
    )
    def test_parses_expected_tokens(self, raw: str, expected: list[str]) -> None:
        assert _parse_label_cell(raw) == expected

    @pytest.mark.parametrize("bad", ["", "   ", None, float("nan"), ",,,"])
    def test_empty_raises(self, bad) -> None:
        with pytest.raises(ValueError, match="empty"):
            _parse_label_cell(bad)


class TestNewSchemaSingleLabel:
    def test_single_label_returns_one_hot(self, tmp_path, default_config, fake_loader):
        manifest = _write_manifest(
            tmp_path,
            [{"file_path": "f.wav", "labels": "fluent", "duration_sec": 1.0}],
        )
        ds = StutteringDataset(
            manifest, default_config, mode="val", audio_loader=fake_loader
        )
        item = ds[0]
        expected = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        expected[LABEL2ID["fluent"]] = 1.0
        assert torch.equal(item["labels"], expected)


class TestNewSchemaMultiLabel:
    def test_multiple_labels_multi_hot(self, tmp_path, default_config, fake_loader):
        manifest = _write_manifest(
            tmp_path,
            [
                {
                    "file_path": "x.wav",
                    "labels": "blocks,prolongations",
                    "duration_sec": 1.0,
                }
            ],
        )
        ds = StutteringDataset(
            manifest, default_config, mode="val", audio_loader=fake_loader
        )
        item = ds[0]
        assert item["labels"][LABEL2ID["blocks"]].item() == 1.0
        assert item["labels"][LABEL2ID["prolongations"]].item() == 1.0
        assert item["labels"][LABEL2ID["fluent"]].item() == 0.0

    def test_all_seven_classes_in_one_row(self, tmp_path, default_config, fake_loader):
        labels_csv = ",".join(CLASS_LABELS)
        manifest = _write_manifest(
            tmp_path,
            [{"file_path": "f.wav", "labels": labels_csv, "duration_sec": 1.0}],
        )
        ds = StutteringDataset(
            manifest, default_config, mode="val", audio_loader=fake_loader
        )
        item = ds[0]
        assert torch.equal(
            item["labels"], torch.ones(NUM_CLASSES, dtype=torch.float32)
        )

    def test_output_shape_always_num_classes(
        self, tmp_path, default_config, fake_loader
    ):
        manifest = _write_manifest(
            tmp_path,
            [
                {"file_path": "a.wav", "labels": "fluent", "duration_sec": 1.0},
                {
                    "file_path": "b.wav",
                    "labels": "blocks,prolongations,word_repetition",
                    "duration_sec": 1.0,
                },
            ],
        )
        ds = StutteringDataset(
            manifest, default_config, mode="val", audio_loader=fake_loader
        )
        for i in range(len(ds)):
            assert ds[i]["labels"].shape == (NUM_CLASSES,)
            assert ds[i]["labels"].dtype == torch.float32


class TestLegacySchemaBackwardCompat:
    def test_legacy_label_column_works(self, tmp_path, default_config, fake_loader):
        manifest = _write_manifest(
            tmp_path,
            [
                {
                    "file_path": "f.wav",
                    "label": "blocks",
                    "label_id": 1,
                    "duration_sec": 1.0,
                    "sample_rate": _TARGET_SR,
                }
            ],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = StutteringDataset(
                manifest, default_config, mode="val", audio_loader=fake_loader
            )
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) >= 1
        assert "legacy single-label schema" in str(deprecations[0].message)
        item = ds[0]
        expected = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        expected[LABEL2ID["blocks"]] = 1.0
        assert torch.equal(item["labels"], expected)


class TestBadRows:
    def test_unknown_label_skips_row_with_warning(
        self, tmp_path, default_config, fake_loader, caplog
    ):
        manifest = _write_manifest(
            tmp_path,
            [
                {"file_path": "ok.wav", "labels": "fluent", "duration_sec": 1.0},
                {"file_path": "bad.wav", "labels": "not_a_class", "duration_sec": 1.0},
            ],
        )
        with caplog.at_level(logging.WARNING):
            ds = StutteringDataset(
                manifest, default_config, mode="val", audio_loader=fake_loader
            )
        assert len(ds) == 1
        assert any("unknown labels" in r.message.lower() for r in caplog.records)

    def test_empty_labels_field_raises(self, tmp_path, default_config, fake_loader):
        manifest = _write_manifest(
            tmp_path,
            [{"file_path": "x.wav", "labels": "", "duration_sec": 1.0}],
        )
        with pytest.raises(ValueError, match="empty"):
            StutteringDataset(
                manifest, default_config, mode="val", audio_loader=fake_loader
            )

    def test_all_rows_skipped_raises(self, tmp_path, default_config, fake_loader):
        manifest = _write_manifest(
            tmp_path,
            [
                {"file_path": "a.wav", "labels": "bogus", "duration_sec": 1.0},
                {"file_path": "b.wav", "labels": "garbage", "duration_sec": 1.0},
            ],
        )
        with pytest.raises(ValueError, match="no usable rows"):
            StutteringDataset(
                manifest, default_config, mode="val", audio_loader=fake_loader
            )


class TestManifestSchemaErrors:
    def test_missing_labels_column_raises(self, tmp_path, default_config, fake_loader):
        manifest = _write_manifest(
            tmp_path, [{"file_path": "x.wav", "duration_sec": 1.0}]
        )
        with pytest.raises(ValueError, match="labels"):
            StutteringDataset(
                manifest, default_config, mode="val", audio_loader=fake_loader
            )

    def test_missing_file_path_column_raises(
        self, tmp_path, default_config, fake_loader
    ):
        manifest = _write_manifest(
            tmp_path, [{"labels": "fluent", "duration_sec": 1.0}]
        )
        with pytest.raises(ValueError, match="file_path"):
            StutteringDataset(
                manifest, default_config, mode="val", audio_loader=fake_loader
            )

    def test_missing_manifest_file_raises(self, tmp_path, default_config, fake_loader):
        nonexistent = str(tmp_path / "does_not_exist.csv")
        with pytest.raises(ValueError, match="not found"):
            StutteringDataset(
                nonexistent, default_config, mode="val", audio_loader=fake_loader
            )

    def test_malformed_csv_raises(self, tmp_path, default_config, fake_loader):
        path = tmp_path / "broken.csv"
        path.write_text('a,b,c\n"unterminated\n', encoding="utf-8")
        with pytest.raises(ValueError):
            StutteringDataset(
                str(path), default_config, mode="val", audio_loader=fake_loader
            )

    def test_invalid_mode_raises(self, tmp_path, default_config, fake_loader):
        manifest = _write_manifest(
            tmp_path, [{"file_path": "f.wav", "labels": "fluent", "duration_sec": 1.0}]
        )
        with pytest.raises(ValueError, match="mode must be one of"):
            StutteringDataset(
                manifest, default_config, mode="predict", audio_loader=fake_loader
            )


class TestAugmentationGating:
    def test_augmentation_only_in_train(
        self, tmp_path, default_config, fake_loader
    ):
        manifest = _write_manifest(
            tmp_path, [{"file_path": "x.wav", "labels": "fluent", "duration_sec": 1.0}]
        )
        calls = {"n": 0}

        def transform(waveform: torch.Tensor) -> torch.Tensor:
            calls["n"] += 1
            return waveform

        train_ds = StutteringDataset(
            manifest,
            default_config,
            augmentation_transforms=transform,
            mode="train",
            audio_loader=fake_loader,
        )
        val_ds = StutteringDataset(
            manifest,
            default_config,
            augmentation_transforms=transform,
            mode="val",
            audio_loader=fake_loader,
        )
        _ = train_ds[0]
        assert calls["n"] == 1
        _ = val_ds[0]
        assert calls["n"] == 1  # unchanged — no augmentation in val


class TestDataLoaderBatching:
    def test_batches_multi_hot_labels(self, tmp_path, default_config, fake_loader):
        manifest = _write_manifest(
            tmp_path,
            [
                {"file_path": "a.wav", "labels": "fluent", "duration_sec": 1.0},
                {"file_path": "b.wav", "labels": "blocks,prolongations", "duration_sec": 1.0},
                {"file_path": "c.wav", "labels": "word_repetition", "duration_sec": 1.0},
                {"file_path": "d.wav", "labels": "interjections", "duration_sec": 1.0},
            ],
        )
        # bypass the public get_dataloader because it doesn't accept audio_loader
        ds = StutteringDataset(
            manifest, default_config, mode="val", audio_loader=fake_loader
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        batches = list(loader)
        assert len(batches) == 2
        first = batches[0]
        assert first["labels"].shape == (2, NUM_CLASSES)
        assert first["labels"].dtype == torch.float32


class TestPreprocessingOverrides:
    def test_alternate_normalize_method_applies(
        self, tmp_path, default_config, fake_loader
    ):
        cfg = {**default_config, "normalize_method": "rms"}
        manifest = _write_manifest(
            tmp_path, [{"file_path": "x.wav", "labels": "fluent", "duration_sec": 1.0}]
        )
        ds = StutteringDataset(manifest, cfg, mode="val", audio_loader=fake_loader)
        # waveform is all zeros from fake_loader; rms normalize must not crash
        item = ds[0]
        assert item["waveform"].shape == (1, _TARGET_SAMPLES)

    def test_alternate_max_duration_applies(
        self, tmp_path, default_config, fake_loader
    ):
        cfg = {**default_config, "max_duration_sec": 5.0}
        manifest = _write_manifest(
            tmp_path, [{"file_path": "x.wav", "labels": "fluent", "duration_sec": 1.0}]
        )
        ds = StutteringDataset(manifest, cfg, mode="val", audio_loader=fake_loader)
        item = ds[0]
        # 5 s @ 16 kHz = 80_000 samples
        assert item["waveform"].shape[-1] == int(5.0 * _TARGET_SR)


class TestDurationDefault:
    def test_missing_duration_defaults_to_zero(
        self, tmp_path, default_config, fake_loader
    ):
        manifest = _write_manifest(
            tmp_path, [{"file_path": "x.wav", "labels": "fluent"}]
        )
        ds = StutteringDataset(
            manifest, default_config, mode="val", audio_loader=fake_loader
        )
        assert ds[0]["duration"] == 0.0


class TestPublicGetDataloader:
    def test_get_dataloader_smoke(
        self, tmp_path, default_config, fake_loader, monkeypatch
    ):
        # patch the audio_loader module so the convenience factory uses our stub
        from ai.preprocessing import audio_loader as audio_mod

        monkeypatch.setattr(audio_mod, "load_audio", fake_loader)
        manifest = _write_manifest(
            tmp_path,
            [
                {"file_path": "a.wav", "labels": "fluent", "duration_sec": 1.0},
                {"file_path": "b.wav", "labels": "blocks", "duration_sec": 1.0},
            ],
        )
        loader = get_dataloader(
            manifest_path=manifest,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            preprocessing_config=default_config,
            mode="val",
        )
        first = next(iter(loader))
        assert first["labels"].shape == (2, NUM_CLASSES)
