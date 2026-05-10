"""Multi-label audio dataset reading a CSV manifest of file paths + class lists.

New manifest schema (v2 — multi-label):
    file_path, labels, duration_sec, sample_rate

where ``labels`` is a comma-separated list of class names from
``shared.labels.CLASS_LABELS``. Examples:
    "fluent"
    "blocks,prolongations"
    "part_word_repetition"

Legacy schema (v1 — single-label, backward-compatible read-only):
    file_path, label, label_id, duration_sec, sample_rate

When a v1 manifest is detected, each row's single ``label`` is treated as a
one-element list and a DeprecationWarning is emitted exactly once per dataset
instance.

Row hygiene:
    * Empty ``labels`` field on any row raises ValueError. There is no sensible
      multi-hot for "this clip has no annotation".
    * If any token in ``labels`` is not in CLASS_LABELS, the row is logged and
      skipped. The dataset never crashes on bad data mid-epoch; it crashes
      eagerly at construction if every row was bad.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ai.preprocessing import audio_loader as _audio_loader_module
from ai.preprocessing.audio_loader import (
    TARGET_DURATION_SEC,
    normalize_waveform,
    pad_or_truncate,
)
from shared.labels import CLASS_LABELS, get_multi_hot

logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"train", "val", "test"})
_LABELS_COLUMN_NEW = "labels"
_LABEL_COLUMN_LEGACY = "label"
_PATH_COLUMN = "file_path"
_DURATION_COLUMN = "duration_sec"

# Audio-loader callable allows tests to inject a stub without touching disk.
AudioLoader = Callable[..., tuple[torch.Tensor, int]]


def _parse_label_cell(cell: object) -> list[str]:
    """Parse one manifest cell into a list of class-name strings.

    Args:
        cell: The raw value from the labels column (str, NaN, etc.).

    Returns:
        List of stripped, non-empty label tokens.

    Raises:
        ValueError: If the cell is missing, NaN, or contains only whitespace.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        raise ValueError("labels field is empty")
    text = str(cell).strip()
    if not text:
        raise ValueError("labels field is empty")
    tokens = [t.strip() for t in text.split(",")]
    tokens = [t for t in tokens if t]
    if not tokens:
        raise ValueError("labels field is empty")
    return tokens


class StutteringDataset(Dataset):
    """Multi-label stuttering classification dataset.

    Each ``__getitem__`` returns a dict with:
        waveform   : 1-D float32 tensor (resampled, normalized, padded)
        labels     : 1-D float32 multi-hot tensor of shape (NUM_CLASSES,)
        file_path  : str — for debugging
        duration   : float — original clip duration in seconds
    """

    def __init__(
        self,
        manifest_csv_path: str,
        preprocessing_config: dict,
        augmentation_transforms: Optional[Callable] = None,
        mode: str = "train",
        audio_loader: Optional[AudioLoader] = None,
    ) -> None:
        """Build the dataset by parsing the manifest and resolving labels.

        Args:
            manifest_csv_path: Path to a CSV with the v2 (or legacy v1) schema.
            preprocessing_config: Dict with keys ``target_sr``,
                ``max_duration_sec``, ``normalize_method``, ``trim_silence``.
            augmentation_transforms: Optional callable applied to the waveform
                when ``mode == 'train'``.
            mode: One of ``"train"``, ``"val"``, ``"test"``.
            audio_loader: Injection point for ``load_audio`` (tests pass a stub).

        Raises:
            ValueError: If ``mode`` is invalid, the manifest can't be read,
                or every row is unusable (all skipped).
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}"
            )
        df = self._read_manifest(manifest_csv_path)
        self._rows, used_legacy = self._extract_rows(df, manifest_csv_path)
        if used_legacy:
            warnings.warn(
                f"Manifest {manifest_csv_path} uses legacy single-label schema "
                f"({_LABEL_COLUMN_LEGACY!r} column). Treating each row's label "
                f"as a one-element list. Migrate to a {_LABELS_COLUMN_NEW!r} "
                f"column (comma-separated) before training new models.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._mode = mode
        self._transforms = augmentation_transforms
        # Resolve lazily so monkeypatching ``audio_loader.load_audio`` works
        # even when callers don't pass the loader explicitly.
        self._audio_loader = audio_loader if audio_loader is not None else _audio_loader_module.load_audio
        self._target_sr: int = int(preprocessing_config["target_sr"])
        self._max_dur: float = float(preprocessing_config["max_duration_sec"])
        self._norm_method: str = str(preprocessing_config["normalize_method"])
        self._trim_silence: bool = bool(preprocessing_config["trim_silence"])

        logger.debug(
            "StutteringDataset ready: %d samples, mode=%s, legacy=%s",
            len(self._rows),
            mode,
            used_legacy,
        )

    @staticmethod
    def _read_manifest(path: str) -> pd.DataFrame:
        """Read the manifest CSV, raising ValueError on read failure."""
        try:
            return pd.read_csv(path)
        except FileNotFoundError as exc:
            raise ValueError(f"Manifest not found: {path}") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"Manifest is not valid CSV: {path}") from exc

    @staticmethod
    def _extract_rows(
        df: pd.DataFrame, manifest_path: str
    ) -> tuple[list[dict], bool]:
        """Walk the dataframe and produce a list of usable rows.

        Returns:
            (rows, used_legacy) where ``rows`` is the kept rows and
            ``used_legacy`` is True if the labels came from the v1 column.

        Raises:
            ValueError: If the manifest has neither ``labels`` nor ``label``
                column, or if every row was dropped.
        """
        if _PATH_COLUMN not in df.columns:
            raise ValueError(
                f"Manifest must contain a {_PATH_COLUMN!r} column; "
                f"got: {list(df.columns)}"
            )

        has_new = _LABELS_COLUMN_NEW in df.columns
        has_legacy = _LABEL_COLUMN_LEGACY in df.columns
        if not has_new and not has_legacy:
            raise ValueError(
                f"Manifest must contain either {_LABELS_COLUMN_NEW!r} (multi-label) "
                f"or {_LABEL_COLUMN_LEGACY!r} (single-label) column; "
                f"got: {list(df.columns)}"
            )
        label_col = _LABELS_COLUMN_NEW if has_new else _LABEL_COLUMN_LEGACY
        used_legacy = not has_new

        rows: list[dict] = []
        for idx, row in df.iterrows():
            file_path = str(row[_PATH_COLUMN]).strip()
            try:
                labels = _parse_label_cell(row[label_col])
            except ValueError as exc:
                raise ValueError(
                    f"{manifest_path} row {idx}: {exc} (file_path={file_path!r})"
                ) from exc
            unknown = [t for t in labels if t not in CLASS_LABELS]
            if unknown:
                logger.warning(
                    "Skipping %s row %d (file_path=%s): unknown labels %s",
                    manifest_path,
                    idx,
                    file_path,
                    unknown,
                )
                continue
            duration = (
                float(row[_DURATION_COLUMN])
                if _DURATION_COLUMN in df.columns and pd.notna(row[_DURATION_COLUMN])
                else 0.0
            )
            rows.append(
                {"file_path": file_path, "labels": labels, "duration": duration}
            )

        if not rows:
            raise ValueError(
                f"Manifest {manifest_path} contains no usable rows "
                f"(all rows had unknown labels or empty fields)"
            )
        return rows, used_legacy

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        """Load one item: waveform tensor + multi-hot label tensor."""
        row = self._rows[idx]
        training_flag = self._mode == "train" and self._trim_silence
        waveform, _ = self._audio_loader(
            row["file_path"], target_sr=self._target_sr, training=training_flag
        )

        if self._norm_method != "peak":
            waveform = normalize_waveform(waveform, method=self._norm_method)
        if self._max_dur != TARGET_DURATION_SEC:
            waveform = pad_or_truncate(waveform, self._target_sr, self._max_dur)
        if self._mode == "train" and self._transforms is not None:
            waveform = self._transforms(waveform)

        multi_hot = get_multi_hot(row["labels"])
        return {
            "waveform": waveform,
            "labels": multi_hot,
            "file_path": row["file_path"],
            "duration": row["duration"],
        }


def get_dataloader(
    manifest_path: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    preprocessing_config: dict,
    augmentation_transforms: Optional[Callable] = None,
    mode: str = "train",
) -> DataLoader:
    """Convenience factory: wrap StutteringDataset in a torch DataLoader."""
    ds = StutteringDataset(
        manifest_path, preprocessing_config, augmentation_transforms, mode
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
