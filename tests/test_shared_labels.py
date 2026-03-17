"""Tests for the shared 7-class taxonomy mapping."""

from shared.labels import CLASS_LABELS, ID2LABEL, LABEL2ID, NUM_CLASSES


def test_label_count_and_order():
    assert NUM_CLASSES == 7
    assert CLASS_LABELS[0] == "fluent"
    assert CLASS_LABELS[-1] == "word_repetition"


def test_bidirectional_mapping_is_consistent():
    assert len(LABEL2ID) == NUM_CLASSES
    assert len(ID2LABEL) == NUM_CLASSES
    for label, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == label

