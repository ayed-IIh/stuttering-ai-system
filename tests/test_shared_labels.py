"""Tests for shared.labels — multi-hot encoding and taxonomy invariants."""

from __future__ import annotations

import pytest
import torch

from shared.labels import (
    CLASS_LABELS,
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    get_multi_hot,
)


class TestTaxonomyInvariants:
    """Sanity checks that the constants remain internally consistent."""

    def test_num_classes_matches_class_labels(self) -> None:
        assert NUM_CLASSES == len(CLASS_LABELS)

    def test_label2id_round_trips(self) -> None:
        for label in CLASS_LABELS:
            assert ID2LABEL[LABEL2ID[label]] == label

    def test_class_labels_unique(self) -> None:
        assert len(set(CLASS_LABELS)) == len(CLASS_LABELS)


class TestGetMultiHotShape:
    """``get_multi_hot`` must always return shape (NUM_CLASSES,) float32."""

    @pytest.mark.parametrize(
        "labels",
        [
            [],
            ["fluent"],
            list(CLASS_LABELS),
            ["blocks", "prolongations"],
            ["blocks", "blocks"],  # duplicate
        ],
    )
    def test_shape_is_always_num_classes(self, labels: list[str]) -> None:
        out = get_multi_hot(labels)
        assert out.shape == (NUM_CLASSES,)

    @pytest.mark.parametrize(
        "labels",
        [
            [],
            ["fluent"],
            list(CLASS_LABELS),
        ],
    )
    def test_dtype_is_float32(self, labels: list[str]) -> None:
        out = get_multi_hot(labels)
        assert out.dtype == torch.float32


class TestGetMultiHotValues:
    """Values must be 1.0 at present indices, 0.0 elsewhere."""

    def test_single_label(self) -> None:
        out = get_multi_hot(["fluent"])
        expected = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        expected[LABEL2ID["fluent"]] = 1.0
        assert torch.equal(out, expected)

    @pytest.mark.parametrize(
        "labels,expected_indices",
        [
            (
                ["blocks", "prolongations"],
                [LABEL2ID["blocks"], LABEL2ID["prolongations"]],
            ),
            (
                ["fluent", "word_repetition"],
                [LABEL2ID["fluent"], LABEL2ID["word_repetition"]],
            ),
            (
                ["part_word_repetition", "phrase_repetition", "word_repetition"],
                [
                    LABEL2ID["part_word_repetition"],
                    LABEL2ID["phrase_repetition"],
                    LABEL2ID["word_repetition"],
                ],
            ),
        ],
    )
    def test_multiple_labels(
        self, labels: list[str], expected_indices: list[int]
    ) -> None:
        out = get_multi_hot(labels)
        for idx in range(NUM_CLASSES):
            if idx in expected_indices:
                assert out[idx].item() == 1.0
            else:
                assert out[idx].item() == 0.0

    def test_all_classes(self) -> None:
        out = get_multi_hot(list(CLASS_LABELS))
        assert torch.equal(out, torch.ones(NUM_CLASSES, dtype=torch.float32))

    def test_empty_list_returns_all_zeros(self) -> None:
        out = get_multi_hot([])
        assert torch.equal(out, torch.zeros(NUM_CLASSES, dtype=torch.float32))

    def test_duplicate_labels_are_idempotent(self) -> None:
        single = get_multi_hot(["blocks"])
        duplicated = get_multi_hot(["blocks", "blocks", "blocks"])
        assert torch.equal(single, duplicated)

    def test_accepts_tuple_input(self) -> None:
        from_list = get_multi_hot(["blocks", "prolongations"])
        from_tuple = get_multi_hot(("blocks", "prolongations"))
        assert torch.equal(from_list, from_tuple)


class TestGetMultiHotErrors:
    """Invalid inputs must raise ValueError with a useful message."""

    @pytest.mark.parametrize(
        "bad_label",
        ["", "FLUENT", "Repetitions", "unknown_class", "fluent ", " blocks"],
    )
    def test_invalid_class_name_raises(self, bad_label: str) -> None:
        with pytest.raises(ValueError, match="Unknown class label"):
            get_multi_hot([bad_label])

    def test_error_message_lists_valid_labels(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            get_multi_hot(["not_a_class"])
        msg = str(exc_info.value)
        for label in CLASS_LABELS:
            assert label in msg

    def test_invalid_label_among_valid_ones_still_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown class label: 'bogus'"):
            get_multi_hot(["fluent", "bogus", "blocks"])
