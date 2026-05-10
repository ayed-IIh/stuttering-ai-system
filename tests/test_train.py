"""Tests for ai.training.train — pure helpers and the multi-label decode path.

Note: ``run_training`` end-to-end is not exercised here because it requires
downloading a real Wav2Vec2 checkpoint and reading real audio files. The
behavior is tested via its decomposable helpers: ``_decode_multi_hot``,
``_validation_metrics``, ``_read_threshold``, ``_build_scheduler``,
and the ``_LOG_FIELDS`` schema constant.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score, hamming_loss
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from ai.training import train as train_mod
from ai.training.train import (
    _LOG_FIELDS,
    _build_scheduler,
    _decode_multi_hot,
    _read_threshold,
    _validation_metrics,
)
from ai.training.checkpoint_utils import append_training_log_csv
from shared.labels import NUM_CLASSES


class TestDecodeMultiHot:
    """`_decode_multi_hot` is the single-source-of-truth decoder."""

    def _logits(self, values: list[list[float]]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32)

    def test_uses_sigmoid_not_softmax(self) -> None:
        # logits all the same value → sigmoid is identical for all → either all
        # above OR all below threshold, never just one (which is what argmax would do).
        logits = self._logits([[0.0] * NUM_CLASSES])
        preds, probs = _decode_multi_hot(logits, threshold=0.5)
        # sigmoid(0)=0.5, exactly threshold → preds all 1.0
        assert preds.shape == (1, NUM_CLASSES)
        np.testing.assert_array_almost_equal(probs[0], np.full(NUM_CLASSES, 0.5))

    def test_threshold_zero_activates_all_classes(self) -> None:
        # any logit gives sigmoid > 0 → predicted at threshold 0.0
        logits = self._logits([[-10.0] * NUM_CLASSES])
        preds, _ = _decode_multi_hot(logits, threshold=0.0)
        assert preds.sum() == NUM_CLASSES

    def test_threshold_one_activates_no_classes(self) -> None:
        # sigmoid output is in (0, 1) — strictly less than 1.0, so threshold=1.0
        # produces an all-zero prediction.
        logits = self._logits([[10.0] * NUM_CLASSES])
        preds, _ = _decode_multi_hot(logits, threshold=1.0)
        assert preds.sum() == 0.0

    def test_threshold_at_boundary(self) -> None:
        # sigmoid(0) == 0.5 exactly. With threshold 0.5 (>=), all classes fire.
        logits = self._logits([[0.0] * NUM_CLASSES])
        preds_eq, _ = _decode_multi_hot(logits, threshold=0.5)
        # threshold slightly above 0.5 → no class fires
        preds_above, _ = _decode_multi_hot(logits, threshold=0.5001)
        assert preds_eq.sum() == NUM_CLASSES
        assert preds_above.sum() == 0.0

    def test_exactly_one_class_above_threshold(self) -> None:
        # craft logits so only index 1 has sigmoid above 0.5
        logit_row = [-10.0] * NUM_CLASSES
        logit_row[1] = 10.0
        preds, _ = _decode_multi_hot(self._logits([logit_row]), threshold=0.5)
        assert preds[0][1] == 1.0
        assert preds[0].sum() == 1.0

    def test_all_seven_classes_above_threshold(self) -> None:
        preds, _ = _decode_multi_hot(self._logits([[10.0] * NUM_CLASSES]), threshold=0.5)
        assert preds.sum() == NUM_CLASSES

    def test_returns_numpy_arrays(self) -> None:
        preds, probs = _decode_multi_hot(self._logits([[0.0] * NUM_CLASSES]), threshold=0.5)
        assert isinstance(preds, np.ndarray)
        assert isinstance(probs, np.ndarray)
        assert preds.dtype == np.float32 or preds.dtype == np.float64

    def test_bad_logit_shape_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Bad logits shape"):
            _decode_multi_hot(torch.zeros(NUM_CLASSES), threshold=0.5)
        with pytest.raises(RuntimeError, match="Bad logits shape"):
            _decode_multi_hot(torch.zeros((4, NUM_CLASSES + 1)), threshold=0.5)


class TestValidationMetrics:
    """Multi-label metrics must use the correct sklearn averaging modes."""

    def test_perfect_predictions(self) -> None:
        # Every class must have at least one positive in y_true, otherwise sklearn
        # treats it as undefined and zero_division=0 drags macro-F1 below 1.0.
        y = np.eye(NUM_CLASSES, dtype=np.float32)
        m = _validation_metrics(y, y)
        assert m["macro_f1"] == 1.0
        assert m["sample_f1"] == 1.0
        assert m["hamming_loss"] == 0.0
        assert m["exact_match_accuracy"] == 1.0

    def test_all_wrong_predictions(self) -> None:
        y_true = np.ones((2, NUM_CLASSES), dtype=np.float32)
        y_pred = np.zeros((2, NUM_CLASSES), dtype=np.float32)
        m = _validation_metrics(y_true, y_pred)
        assert m["macro_f1"] == 0.0
        assert m["sample_f1"] == 0.0
        assert m["hamming_loss"] == 1.0
        assert m["exact_match_accuracy"] == 0.0

    def test_matches_sklearn_directly(self) -> None:
        rng = np.random.default_rng(0)
        y_true = (rng.random((10, NUM_CLASSES)) > 0.5).astype(np.float32)
        y_pred = (rng.random((10, NUM_CLASSES)) > 0.5).astype(np.float32)
        m = _validation_metrics(y_true, y_pred)
        np.testing.assert_almost_equal(
            m["macro_f1"], f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        np.testing.assert_almost_equal(
            m["sample_f1"], f1_score(y_true, y_pred, average="samples", zero_division=0)
        )
        np.testing.assert_almost_equal(m["hamming_loss"], hamming_loss(y_true, y_pred))

    def test_empty_arrays_handled(self) -> None:
        y = np.empty((0, NUM_CLASSES))
        m = _validation_metrics(y, y)
        assert m["macro_f1"] == 0.0
        assert m["hamming_loss"] == 0.0


class TestReadThreshold:
    """``_read_threshold`` validates the config bounds."""

    @pytest.mark.parametrize("v", [0.0, 0.25, 0.5, 0.99, 1.0])
    def test_accepts_valid_range(self, v: float) -> None:
        assert _read_threshold({"threshold": v}) == v

    @pytest.mark.parametrize("v", [-0.01, 1.01, -1.0, 2.0])
    def test_rejects_out_of_range(self, v: float) -> None:
        with pytest.raises(ValueError, match="threshold must be"):
            _read_threshold({"threshold": v})

    def test_missing_key_raises(self) -> None:
        with pytest.raises(KeyError, match="threshold"):
            _read_threshold({})


class TestBuildScheduler:
    """``_build_scheduler`` dispatches to the cosine vs linear variants."""

    def _optimizer(self):
        param = torch.nn.Parameter(torch.zeros(1))
        return AdamW([param], lr=1e-4)

    def test_linear_default(self) -> None:
        sch = _build_scheduler("linear", self._optimizer(), 0, 10)
        # No clean isinstance check (transformers returns LambdaLR for both),
        # but verify the function returns a non-None scheduler with .step().
        assert sch is not None
        sch.step()

    def test_cosine(self) -> None:
        sch = _build_scheduler("cosine", self._optimizer(), 1, 5)
        assert sch is not None
        sch.step()

    def test_unknown_falls_back_to_linear(self) -> None:
        sch = _build_scheduler("bogus", self._optimizer(), 0, 5)
        assert sch is not None
        sch.step()


class TestLogFields:
    """The log schema must contain every multi-label metric."""

    def test_required_fields_present(self) -> None:
        for f in (
            "epoch",
            "train_loss",
            "val_loss",
            "val_macro_f1",
            "val_sample_f1",
            "val_hamming_loss",
            "threshold",
        ):
            assert f in _LOG_FIELDS

    def test_field_order_stable(self) -> None:
        # epoch must come first for readability
        assert _LOG_FIELDS[0] == "epoch"


class TestLogCsvWriteRoundTrip:
    """End-to-end: append_training_log_csv writes the expected header + row."""

    def test_writes_all_required_columns(self, tmp_path: Path) -> None:
        log_path = tmp_path / "training_log.csv"
        row = {
            "epoch": 1,
            "train_loss": "0.500000",
            "val_loss": "0.400000",
            "val_exact_match_accuracy": "0.700000",
            "val_macro_f1": "0.650000",
            "val_sample_f1": "0.700000",
            "val_hamming_loss": "0.100000",
            "learning_rate": "1.00000000e-04",
            "threshold": "0.5000",
        }
        append_training_log_csv(log_path, _LOG_FIELDS, row)
        with log_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        for f in _LOG_FIELDS:
            assert f in rows[0]


class TestTrainOneEpochUsesBce:
    """Smoke-test that ``_train_one_epoch`` uses BCE-with-logits, not CE."""

    def test_bce_called(self, monkeypatch) -> None:
        called: dict[str, int] = {"bce": 0, "ce": 0}

        def fake_bce(logits, labels):  # noqa: ANN001
            called["bce"] += 1
            return torch.tensor(0.0, requires_grad=True)

        def fake_ce(logits, labels):  # noqa: ANN001
            called["ce"] += 1
            return torch.tensor(0.0, requires_grad=True)

        monkeypatch.setattr(train_mod.F, "binary_cross_entropy_with_logits", fake_bce)
        monkeypatch.setattr(train_mod.F, "cross_entropy", fake_ce)

        # Minimal fake model: linear projection from input_values mean → NUM_CLASSES
        class _M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(1, NUM_CLASSES)

            def train(self, mode: bool = True):  # type: ignore[override]
                self.lin.train(mode)
                return self

            def parameters(self, recurse: bool = True):  # type: ignore[override]
                return self.lin.parameters()

            def __call__(self, input_values, attention_mask):  # noqa: ANN001
                return self.lin(input_values.mean(dim=1, keepdim=True))

        model = _M()
        device = torch.device("cpu")
        optimizer = AdamW(model.parameters(), lr=1e-4)
        sch = _build_scheduler("linear", optimizer, 0, 2)

        batch = {
            "input_values": torch.randn(2, 4),
            "attention_mask": torch.ones(2, 4),
            "labels": torch.zeros(2, NUM_CLASSES),
        }

        train_mod._train_one_epoch(
            model,
            [batch, batch],
            optimizer,
            sch,
            device,
            max_grad_norm=1.0,
        )

        assert called["bce"] == 2  # one per batch
        assert called["ce"] == 0
