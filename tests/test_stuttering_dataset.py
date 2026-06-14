import math

import pandas as pd
import pytest
import torch
import torchaudio

from ai.dataset.stuttering_dataset import StutteringDataset, get_dataloader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path, num_frames: int = 16_000, sr: int = 16_000) -> str:
    t = torch.linspace(0, num_frames / sr, num_frames)
    waveform = torch.sin(2 * math.pi * 440 * t).unsqueeze(0)
    torchaudio.save(str(path), waveform.float(), sr)
    return str(path)


def _write_manifest(tmp_path, rows: list) -> str:
    p = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(str(p), index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    return {
        "target_sr": 16_000,
        "max_duration_sec": 10.0,
        "normalize_method": "peak",
        "trim_silence": False,
    }


@pytest.fixture
def sample_manifest(tmp_path):
    rows = []
    for label, label_id in [
        ("fluent", 0), ("blocks", 1), ("prolongations", 3)
    ]:
        wav = tmp_path / f"{label}.wav"
        _make_wav(wav)
        rows.append({
            "file_path": str(wav),
            "label": label,
            "label_id": label_id,
            "duration_sec": 1.0,
            "sample_rate": 16_000,
        })
    return _write_manifest(tmp_path, rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dataset_length_matches_manifest(sample_manifest, default_config):
    ds = StutteringDataset(sample_manifest, default_config)
    expected = len(pd.read_csv(sample_manifest))
    assert len(ds) == expected


def test_getitem_returns_correct_structure(sample_manifest, default_config):
    ds = StutteringDataset(sample_manifest, default_config)
    item = ds[0]

    assert set(item.keys()) == {"waveform", "label", "file_path", "duration"}
    assert isinstance(item["waveform"], torch.Tensor)
    assert item["waveform"].dtype == torch.float32
    assert item["waveform"].shape == (1, 160_000)
    assert isinstance(item["label"], int)
    assert isinstance(item["file_path"], str)
    assert isinstance(item["duration"], float)


def test_label_mappings():
    expected = {
        "fluent": 0,
        "blocks": 1,
        "interjections": 2,
        "prolongations": 3,
        "part_word_repetition": 4,
        "phrase_repetition": 5,
        "word_repetition": 6,
    }
    assert StutteringDataset.LABEL2ID == expected
    assert StutteringDataset.ID2LABEL == {v: k for k, v in expected.items()}
    assert len(StutteringDataset.LABEL2ID) == 7
    assert len(StutteringDataset.ID2LABEL) == 7


def test_dataloader_full_epoch(sample_manifest, default_config):
    loader = get_dataloader(
        manifest_path=sample_manifest,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        preprocessing_config=default_config,
        augmentation_transforms=None,
        mode="train",
    )
    batches = list(loader)

    assert len(batches) > 0
    first = batches[0]
    assert first["waveform"].ndim == 3      # (batch, 1, 160000)
    # DataLoader collates Python int → LongTensor
    assert first["label"].dtype == torch.int64


def test_augmentation_applied_in_train_only(sample_manifest, default_config):
    # count invocations — value checks are fragile on sine waves (negated sine
    # still has a positive max from its original negative lobe)
    class _Counter:
        def __init__(self):
            self.n = 0

        def __call__(self, w):
            self.n += 1
            return w

    counter = _Counter()
    train_ds = StutteringDataset(
        sample_manifest, default_config,
        augmentation_transforms=counter, mode="train"
    )
    val_ds = StutteringDataset(
        sample_manifest, default_config,
        augmentation_transforms=counter, mode="val"
    )

    train_ds[0]
    assert counter.n == 1, "transform should fire once for train"

    val_ds[0]
    assert counter.n == 1, "transform must not fire for val"


def test_label_id_matches_manifest_row(sample_manifest, default_config):
    df = pd.read_csv(sample_manifest)
    ds = StutteringDataset(sample_manifest, default_config)

    for i in range(len(ds)):
        assert ds[i]["label"] == int(df.iloc[i]["label_id"])


def test_invalid_mode_raises(sample_manifest, default_config):
    with pytest.raises(ValueError, match="mode must be one of"):
        StutteringDataset(sample_manifest, default_config, mode="predict")
