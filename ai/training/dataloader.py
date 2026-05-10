"""DataLoader construction for multi-label audio classification training.

Manifest schema:
    path/file_path/absolute_path  : file location (resolved relative to manifest)
    labels                        : comma-separated class names (multi-label)
                                    OR (legacy) label/class_label/class — single string

Returns multi-hot float32 label tensors of shape ``(NUM_CLASSES,)`` per item.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor

from shared.labels import CLASS_LABELS, NUM_CLASSES, get_multi_hot

logger = logging.getLogger(__name__)

_PATH_COLUMNS = ("path", "file_path", "absolute_path")
_LABELS_COLUMN_NEW = "labels"
_LEGACY_LABEL_COLUMNS = ("label", "class_label", "class")


def _pick_path_column(df: pd.DataFrame) -> str:
    """Return the first present path column, else raise ValueError."""
    for col in _PATH_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        f"Manifest must contain one of {_PATH_COLUMNS}; got: {list(df.columns)}"
    )


def _parse_labels_cell(cell: object) -> list[str]:
    """Split a labels cell into a clean list of tokens.

    Args:
        cell: Raw cell value (str, NaN, etc.).

    Returns:
        Stripped, non-empty label tokens.

    Raises:
        ValueError: If cell is missing or contains only whitespace/commas.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        raise ValueError("labels field is empty")
    text = str(cell).strip()
    if not text:
        raise ValueError("labels field is empty")
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if not tokens:
        raise ValueError("labels field is empty")
    return tokens


def _extract_label_tensors(
    df: pd.DataFrame, manifest_path: Path
) -> tuple[list[torch.Tensor], list[int]]:
    """Parse the manifest's label column(s) into per-row multi-hot tensors.

    Returns:
        (label_tensors, kept_indices) where kept_indices indexes back into ``df``
        for the rows whose labels parsed cleanly.

    Raises:
        ValueError: If no usable label column exists, an empty cell is found,
            or every row was dropped because of unknown class names.
    """
    has_new = _LABELS_COLUMN_NEW in df.columns
    has_legacy = any(c in df.columns for c in _LEGACY_LABEL_COLUMNS)
    if not has_new and not has_legacy:
        raise ValueError(
            f"Manifest must contain {_LABELS_COLUMN_NEW!r} (multi-label) "
            f"or one of {_LEGACY_LABEL_COLUMNS} (single-label); "
            f"got: {list(df.columns)}"
        )

    if has_new:
        label_col = _LABELS_COLUMN_NEW
    else:
        for cand in _LEGACY_LABEL_COLUMNS:
            if cand in df.columns:
                label_col = cand
                break
        warnings.warn(
            f"Manifest {manifest_path} uses legacy single-label column "
            f"{label_col!r}. Treating each row as a one-element list. "
            f"Migrate to a {_LABELS_COLUMN_NEW!r} column (comma-separated).",
            DeprecationWarning,
            stacklevel=2,
        )

    label_tensors: list[torch.Tensor] = []
    kept_indices: list[int] = []
    for idx, raw in enumerate(df[label_col].tolist()):
        try:
            tokens = _parse_labels_cell(raw)
        except ValueError as exc:
            raise ValueError(
                f"{manifest_path} row {idx}: {exc}"
            ) from exc
        unknown = [t for t in tokens if t not in CLASS_LABELS]
        if unknown:
            logger.warning(
                "Skipping %s row %d: unknown labels %s",
                manifest_path,
                idx,
                unknown,
            )
            continue
        label_tensors.append(get_multi_hot(tokens))
        kept_indices.append(idx)

    if not label_tensors:
        raise ValueError(
            f"Manifest {manifest_path} produced no usable rows "
            f"(all labels were unknown)"
        )
    return label_tensors, kept_indices


def _resolve_paths(df: pd.DataFrame, path_col: str, manifest_path: Path) -> list[Path]:
    """Resolve every path in the column relative to the manifest's directory."""
    base = manifest_path.parent.resolve()
    out: list[Path] = []
    for raw in df[path_col].tolist():
        rp = Path(str(raw).strip())
        out.append(rp if rp.is_absolute() else (base / rp).resolve())
    return out


class ManifestAudioDataset(Dataset):
    """Loads mono waveforms and multi-hot label tensors from a CSV manifest."""

    def __init__(
        self,
        manifest_path: Path,
        target_sample_rate: int,
    ) -> None:
        """Build the dataset.

        Args:
            manifest_path: CSV with path + labels columns.
            target_sample_rate: Resample target (Hz).

        Raises:
            ValueError: On schema or label-parsing errors.
        """
        self._manifest_path = manifest_path
        self._target_sample_rate = int(target_sample_rate)

        df = pd.read_csv(manifest_path)
        path_col = _pick_path_column(df)
        all_paths = _resolve_paths(df, path_col, manifest_path)
        label_tensors, kept = _extract_label_tensors(df, manifest_path)
        self._paths = [all_paths[i] for i in kept]
        self._labels = label_tensors

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self._paths[idx]
        labels = self._labels[idx]

        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if sr != self._target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self._target_sample_rate)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        audio = waveform.numpy().astype(np.float32)
        return {"audio": audio, "labels": labels}


def _attention_mask_from_encoded(
    encoded: Any, input_values: torch.Tensor
) -> torch.Tensor:
    """Return attention_mask from the processor output, or a valid full-ones mask."""
    mask: torch.Tensor | None = None
    if isinstance(encoded, dict):
        mask = encoded.get("attention_mask")
    else:
        mask = getattr(encoded, "attention_mask", None)
        if mask is None and hasattr(encoded, "get"):
            mask = encoded.get("attention_mask")
    if mask is not None:
        return mask
    return torch.ones(
        input_values.shape[:2], dtype=torch.long, device=input_values.device
    )


def _collate_wav2vec(
    batch: list[dict[str, Any]],
    processor: Wav2Vec2Processor,
    sampling_rate: int,
) -> dict[str, torch.Tensor]:
    """Collate batched audio + multi-hot label tensors for the model."""
    audios = [item["audio"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    if labels.dim() != 2 or labels.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Bad label batch shape {tuple(labels.shape)}; expected (batch, {NUM_CLASSES})"
        )
    encoded = processor(
        audios,
        sampling_rate=sampling_rate,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_values = encoded["input_values"]
    attention_mask = _attention_mask_from_encoded(encoded, input_values)
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_dataloader(
    manifest_path: str | Path,
    processor: Wav2Vec2Processor,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    label_to_id: dict[str, int] | None = None,
) -> DataLoader:
    """Build a DataLoader over a multi-label CSV manifest.

    Args:
        manifest_path: Path to CSV.
        processor: HuggingFace Wav2Vec2 processor (used for collation).
        batch_size, num_workers, shuffle: standard DataLoader args.
        label_to_id: Accepted for backward compatibility; ignored (labels are
            resolved via :data:`shared.labels.CLASS_LABELS`).

    Returns:
        torch.utils.data.DataLoader.
    """
    _ = label_to_id  # kept in signature so existing callers don't break
    path = Path(manifest_path)
    sr = int(processor.feature_extractor.sampling_rate)
    dataset = ManifestAudioDataset(path, target_sample_rate=sr)

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
