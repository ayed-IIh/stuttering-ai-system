"""
Dataset Split — stuttering-ai-system
=====================================
Reads dataset_inventory.csv and produces stratified train/val/test manifests
at a 70/15/15 ratio across all 7 stuttering classes.

Classes with fewer than MIN_SAMPLES_FOR_SPLIT samples are forced into the
training set entirely (currently: phrase_repetition, 2 samples).

Usage:
  python ai/dataset/split_dataset.py [OPTIONS]

  --inventory-csv PATH   default: ai/dataset/metadata/dataset_inventory.csv
  --output-dir    PATH   default: ai/dataset/processed
  --seed          INT    default: 42

Outputs:
  ai/dataset/processed/train_manifest.csv
  ai/dataset/processed/val_manifest.csv
  ai/dataset/processed/test_manifest.csv
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

# ── Constants ──────────────────────────────────────────────────────────────────

SCRIPT_VERSION = "1.0.0"
RANDOM_SEED = 42

# Keep in sync with STUTTERING_CLASSES in ai/models/stuttering_classifier.py
LABEL_TO_ID: dict[str, int] = {
    "fluent": 0,
    "blocks": 1,
    "interjections": 2,
    "prolongations": 3,
    "part_word_repetition": 4,
    "phrase_repetition": 5,
    "word_repetition": 6,
}
assert len(LABEL_TO_ID) == 7, "LABEL_TO_ID must have exactly 7 entries"

# floor(0.15 * n) >= 1 requires n >= 7; below this the second-pass split breaks
MIN_SAMPLES_FOR_SPLIT = 7
DATA_AUG_WARN_THRESHOLD = 20

_FLAG_COLS = ["is_zero_byte", "is_corrupted", "is_duplicate", "has_missing_label"]

# ── Data models ────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SplitConfig:
    inventory_csv: Path
    output_dir: Path
    random_seed: int = RANDOM_SEED
    min_samples_for_split: int = MIN_SAMPLES_FOR_SPLIT


# ── Core logic ─────────────────────────────────────────────────────────────────


def load_inventory(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # pandas 2.x infers "True"/"False" columns as bool dtype; normalize via str
    present = [c for c in _FLAG_COLS if c in df.columns]
    if present:
        flagged = df[present].apply(
            lambda col: col.astype(str).map({"True": True, "False": False})
        )
        df = df[~flagged.any(axis=1)].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable records in {csv_path}")
    return df


def validate_label_coverage(df: pd.DataFrame) -> None:
    unknown = set(df["class_label"].unique()) - LABEL_TO_ID.keys()
    if unknown:
        raise ValueError(f"Unknown class labels in inventory: {sorted(unknown)}")


def warn_small_classes(df: pd.DataFrame) -> None:
    counts = df["class_label"].value_counts()
    small = counts[counts < DATA_AUG_WARN_THRESHOLD]
    if small.empty:
        return
    print("\nWARNING: the following classes have fewer than 20 total samples.")
    print("         Consider data augmentation before training:")
    for label, n in small.items():
        print(f"           {label}: {n} samples")


def partition_by_threshold(
    df: pd.DataFrame, min_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df["class_label"].value_counts()
    tiny = counts[counts < min_count].index
    forced_mask = df["class_label"].isin(tiny)
    return (
        df[~forced_mask].reset_index(drop=True),
        df[forced_mask].reset_index(drop=True),
    )


def warn_forced_classes(forced_df: pd.DataFrame) -> None:
    if forced_df.empty:
        return
    for label, n in forced_df["class_label"].value_counts().items():
        print(
            f"WARNING: '{label}' has {n} sample(s)"
            f" (< MIN_SAMPLES_FOR_SPLIT={MIN_SAMPLES_FOR_SPLIT})"
            " -- all assigned to train"
        )


def _stratified_split(
    df: pd.DataFrame, config: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    has_speakers = "speaker_id" in df.columns

    if has_speakers:
        # speaker-independent: keeps speakers out of val/test if they appear in train
        splitter1 = GroupShuffleSplit(
            n_splits=1, test_size=0.30, random_state=config.random_seed
        )
        splitter2 = GroupShuffleSplit(
            n_splits=1, test_size=0.50, random_state=config.random_seed
        )
        train_idx, rest_idx = next(splitter1.split(df, groups=df["speaker_id"]))
        rest_df = df.iloc[rest_idx].reset_index(drop=True)
        val_idx, test_idx = next(splitter2.split(rest_df, groups=rest_df["speaker_id"]))
    else:
        splitter1 = StratifiedShuffleSplit(
            n_splits=1, test_size=0.30, random_state=config.random_seed
        )
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, test_size=0.50, random_state=config.random_seed
        )
        train_idx, rest_idx = next(splitter1.split(df, df["class_label"]))
        rest_df = df.iloc[rest_idx].reset_index(drop=True)
        val_idx, test_idx = next(splitter2.split(rest_df, rest_df["class_label"]))

    return (
        df.iloc[train_idx].reset_index(drop=True),
        rest_df.iloc[val_idx].reset_index(drop=True),
        rest_df.iloc[test_idx].reset_index(drop=True),
    )


def split_dataset(
    splittable: pd.DataFrame,
    forced_train: pd.DataFrame,
    config: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, val_df, test_df = _stratified_split(splittable, config)
    if not forced_train.empty:
        train_df = pd.concat([train_df, forced_train], ignore_index=True)
    return train_df, val_df, test_df


def build_manifest(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_path": df["absolute_path"].values,
            "label": df["class_label"].values,
            "label_id": df["class_label"].map(LABEL_TO_ID).values,
            "duration_sec": df["duration_seconds"].values,
            "sample_rate": df["sample_rate_hz"].values,
        }
    )


def print_split_distribution(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    all_labels = sorted(LABEL_TO_ID.keys())
    forced = set(train["label"]) - set(val["label"]) - set(test["label"])

    # collect per-label counts
    rows = []
    for lbl in all_labels:
        tr = (train["label"] == lbl).sum()
        v = (val["label"] == lbl).sum()
        te = (test["label"] == lbl).sum()
        rows.append((lbl, tr, v, te))

    w = max(len(lbl) for lbl in all_labels) + 2
    sep = "-" * (w + 28)

    print(f"\n{sep}")
    print(f"  split_dataset v{SCRIPT_VERSION}  --  "
          f"train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"  {'class':<{w}}  train   val  test")
    print(f"  {'-' * (w + 24)}")
    for lbl, tr, v, te in rows:
        note = "  [forced-train]" if lbl in forced else ""
        print(f"  {lbl:<{w}}  {tr:>5}  {v:>5}  {te:>5}{note}")
    print(sep)


def write_manifests(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
    for name, df in splits.items():
        out = output_dir / f"{name}_manifest.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    p = argparse.ArgumentParser(
        description=(
            "Produce stratified train/val/test manifests for the stuttering dataset."
        )
    )
    p.add_argument(
        "--inventory-csv",
        type=Path,
        default=root / "ai/dataset/metadata/dataset_inventory.csv",
        metavar="PATH",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "ai/dataset/processed",
        metavar="PATH",
    )
    p.add_argument("--seed", type=int, default=RANDOM_SEED, metavar="INT")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = SplitConfig(
        inventory_csv=args.inventory_csv,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    df = load_inventory(config.inventory_csv)
    validate_label_coverage(df)
    warn_small_classes(df)

    splittable, forced_train = partition_by_threshold(df, config.min_samples_for_split)
    warn_forced_classes(forced_train)

    train_df, val_df, test_df = split_dataset(splittable, forced_train, config)

    train_m = build_manifest(train_df)
    val_m = build_manifest(val_df)
    test_m = build_manifest(test_df)

    print_split_distribution(train_m, val_m, test_m)
    write_manifests(train_m, val_m, test_m, config.output_dir)


if __name__ == "__main__":
    main()
