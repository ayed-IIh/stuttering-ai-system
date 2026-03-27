"""DataLoader construction for audio classification training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor


_PATH_COLUMNS = ("path", "file_path", "absolute_path")
_LABEL_COLUMNS = ("label", "class_label", "class")


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Manifest must contain one of {candidates}; got columns: {list(df.columns)}"
    )


class ManifestAudioDataset(Dataset):
    """Loads mono 16 kHz waveforms and integer class indices from a CSV manifest."""

    def __init__(
        self,
        manifest_path: Path,
        label_to_id: dict[str, int],
        target_sample_rate: int,
    ) -> None:
        self._manifest_path = manifest_path
        self._label_to_id = label_to_id
        self._target_sample_rate = target_sample_rate

        df = pd.read_csv(manifest_path)
        path_col = _pick_column(df, _PATH_COLUMNS)
        label_col = _pick_column(df, _LABEL_COLUMNS)

        base_dir = manifest_path.parent.resolve()
        resolved: list[Path] = []
        for p in df[path_col].tolist():
            raw_path = Path(str(p).strip())
            if raw_path.is_absolute():
                resolved.append(raw_path)
            else:
                resolved.append((base_dir / raw_path).resolve())
        self._paths = resolved
        self._labels: list[int] = []
        for raw in df[label_col].tolist():
            key = str(raw).strip()
            if key not in label_to_id:
                raise KeyError(f"Unknown label {key!r} in {manifest_path}")
            self._labels.append(label_to_id[key])

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self._paths[idx]
        label = self._labels[idx]

        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if sr != self._target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self._target_sample_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        audio = waveform.numpy().astype(np.float32)
        return {"audio": audio, "label": label}


def _collate_wav2vec(
    batch: list[dict[str, Any]],
    processor: Wav2Vec2Processor,
    sampling_rate: int,
) -> dict[str, torch.Tensor]:
    audios = [item["audio"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    encoded = processor(
        audios,
        sampling_rate=sampling_rate,
        padding=True,
        return_tensors="pt",
    )
    return {
        "input_values": encoded["input_values"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def get_dataloader(
    manifest_path: str | Path,
    processor: Wav2Vec2Processor,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    label_to_id: dict[str, int],
) -> DataLoader:
    """
    Build a DataLoader over a CSV manifest with audio paths and string labels.

    Manifest columns: one of path/file_path/absolute_path, and one of
    label/class_label/class. Labels must match keys in ``label_to_id``.
    """
    path = Path(manifest_path)
    sr = int(processor.feature_extractor.sampling_rate)
    dataset = ManifestAudioDataset(path, label_to_id, target_sample_rate=sr)

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return _collate_wav2vec(batch, processor, sr)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
