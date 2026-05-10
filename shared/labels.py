"""Canonical 7-class label taxonomy shared across the project.

Multi-label semantics: a single audio clip may exhibit zero or more of these
classes simultaneously. ``get_multi_hot`` converts a list of class names into
the multi-hot float32 tensor used as the model target.
"""

from __future__ import annotations

from typing import Iterable

import torch

CLASS_LABELS: tuple[str, ...] = (
    "fluent",
    "blocks",
    "interjections",
    "prolongations",
    "part_word_repetition",
    "phrase_repetition",
    "word_repetition",
)

LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(CLASS_LABELS)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}
NUM_CLASSES: int = len(CLASS_LABELS)


def get_multi_hot(labels: Iterable[str]) -> torch.Tensor:
    """Convert a collection of class names to a multi-hot float32 tensor.

    Duplicates in ``labels`` are idempotent (setting the same index to 1.0
    twice has no extra effect).

    Args:
        labels: Iterable of class names (e.g. list, tuple). Must NOT be a raw
            ``str`` — a single string is iterable and would be split into its
            characters, which is almost certainly not what the caller wants.
            Each item must be a member of ``CLASS_LABELS``. May be empty.

    Returns:
        Float32 tensor of shape ``(NUM_CLASSES,)`` with 1.0 at every present
        class index and 0.0 elsewhere.

    Raises:
        TypeError: If ``labels`` is a raw string.
        ValueError: If any item in ``labels`` is not a member of ``CLASS_LABELS``.
    """
    if isinstance(labels, str):
        raise TypeError(
            "labels must be an iterable of class-name strings (e.g. list/tuple), "
            "not a single str. Wrap a single label as [label]."
        )
    vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for label in labels:
        if label not in LABEL2ID:
            raise ValueError(
                f"Unknown class label: {label!r}. "
                f"Valid labels are: {list(CLASS_LABELS)}"
            )
        vec[LABEL2ID[label]] = 1.0
    return vec
