"""Canonical 7-class label taxonomy shared across the project."""

from __future__ import annotations

CLASS_LABELS = (
    "fluent",
    "blocks",
    "interjections",
    "prolongations",
    "part_word_repetition",
    "phrase_repetition",
    "word_repetition",
)

LABEL2ID = {label: idx for idx, label in enumerate(CLASS_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_CLASSES = len(CLASS_LABELS)

