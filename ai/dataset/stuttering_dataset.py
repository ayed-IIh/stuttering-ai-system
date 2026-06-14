import logging
from typing import Callable, Dict, Optional

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from ai.preprocessing.audio_loader import (
    TARGET_DURATION_SEC,
    load_audio,
    normalize_waveform,
    pad_or_truncate,
)
from shared.labels import ID2LABEL as _ID2LABEL, LABEL2ID as _LABEL2ID

logger = logging.getLogger(__name__)

_VALID_MODES = {"train", "val", "test"}


class StutteringDataset(Dataset):
    # Sourced from the canonical taxonomy (shared.labels), not a local copy.
    LABEL2ID: Dict[str, int] = dict(_LABEL2ID)
    ID2LABEL: Dict[int, str] = dict(_ID2LABEL)

    def __init__(
        self,
        manifest_csv_path: str,
        preprocessing_config: dict,
        augmentation_transforms: Optional[Callable] = None,
        mode: str = "train",
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got '{mode}'"
            )

        self._df = pd.read_csv(manifest_csv_path)
        self._mode = mode
        self._transforms = augmentation_transforms

        # unpack once — avoids repeated dict lookups in __getitem__
        self._target_sr: int = preprocessing_config["target_sr"]
        self._max_dur: float = preprocessing_config["max_duration_sec"]
        self._norm_method: str = preprocessing_config["normalize_method"]
        self._trim_silence: bool = preprocessing_config["trim_silence"]

        logger.debug(
            "StutteringDataset ready: %d samples, mode=%s", len(self._df), mode
        )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict:
        row = self._df.iloc[idx]
        file_path = str(row["file_path"])

        # silence trimming is train-only even when the config enables it —
        # clinical pre-utterance pauses carry diagnostic info at inference
        training_flag = self._mode == "train" and self._trim_silence
        waveform, _ = load_audio(
            file_path, target_sr=self._target_sr, training=training_flag
        )

        # load_audio hard-codes peak normalization and 10 s duration;
        # re-apply if the caller requested something different
        if self._norm_method != "peak":
            waveform = normalize_waveform(waveform, method=self._norm_method)
        if self._max_dur != TARGET_DURATION_SEC:
            waveform = pad_or_truncate(
                waveform, self._target_sr, self._max_dur
            )

        if self._mode == "train" and self._transforms is not None:
            waveform = self._transforms(waveform)

        return {
            "waveform": waveform,
            "label": int(row["label_id"]),       # numpy int64 → Python int
            "file_path": file_path,
            "duration": float(row["duration_sec"]),
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
    # augmentation_transforms must be picklable when num_workers > 0
    ds = StutteringDataset(
        manifest_path, preprocessing_config, augmentation_transforms, mode
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
