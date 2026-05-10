"""Tests for ai.evaluation.evaluate — multi-label metrics + I/O."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai.evaluation.evaluate import (
    _CHART_FILENAME,
    _METRICS_FILENAME,
    _resolve_threshold,
    _validate_threshold,
    compute_metrics,
    decode,
    run_evaluation,
    save_per_class_chart,
)
from shared.labels import CLASS_LABELS, NUM_CLASSES


class TestValidateThreshold:
    @pytest.mark.parametrize("v", [0.0, 0.25, 0.5, 0.99, 1.0])
    def test_accepts_valid(self, v: float) -> None:
        assert _validate_threshold(v) == v

    @pytest.mark.parametrize("v", [-0.01, 1.01, 2.0, -1.0])
    def test_rejects_out_of_range(self, v: float) -> None:
        with pytest.raises(ValueError, match="threshold must be"):
            _validate_threshold(v)


class TestDecode:
    def test_threshold_zero_activates_all(self) -> None:
        probs = np.full((3, NUM_CLASSES), 0.001, dtype=np.float32)
        out = decode(probs, threshold=0.0)
        assert out.sum() == 3 * NUM_CLASSES

    def test_threshold_one_activates_none(self) -> None:
        # sigmoid output is in (0, 1) — never exactly 1.0 in practice.
        probs = np.full((3, NUM_CLASSES), 0.999, dtype=np.float32)
        out = decode(probs, threshold=1.0)
        assert out.sum() == 0.0

    def test_at_boundary_includes(self) -> None:
        probs = np.full((1, NUM_CLASSES), 0.5, dtype=np.float32)
        out = decode(probs, threshold=0.5)
        assert out.sum() == NUM_CLASSES

    @pytest.mark.parametrize(
        "shape",
        [(NUM_CLASSES,), (3, NUM_CLASSES + 1), (3, 4, NUM_CLASSES)],
    )
    def test_bad_shape_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="probs must have shape"):
            decode(np.zeros(shape, dtype=np.float32), threshold=0.5)

    def test_bad_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            decode(np.zeros((1, NUM_CLASSES)), threshold=1.5)


class TestComputeMetricsShape:
    def test_perfect_predictions(self) -> None:
        # Every class has at least one positive — required for non-trivial macro F1.
        y = np.eye(NUM_CLASSES, dtype=np.float32)
        m = compute_metrics(y, y)
        assert m["macro_f1"] == 1.0
        assert m["sample_f1"] == 1.0
        assert m["hamming_loss"] == 0.0
        assert m["accuracy"] == 1.0

    def test_all_wrong_predictions(self) -> None:
        y_true = np.ones((4, NUM_CLASSES), dtype=np.float32)
        y_pred = np.zeros((4, NUM_CLASSES), dtype=np.float32)
        m = compute_metrics(y_true, y_pred)
        assert m["macro_f1"] == 0.0
        assert m["sample_f1"] == 0.0
        assert m["hamming_loss"] == 1.0
        assert m["accuracy"] == 0.0

    def test_empty_arrays_handled(self) -> None:
        empty = np.empty((0, NUM_CLASSES), dtype=np.float32)
        m = compute_metrics(empty, empty)
        assert m["macro_f1"] == 0.0
        assert m["hamming_loss"] == 0.0
        assert set(m["per_class"].keys()) == set(CLASS_LABELS)
        assert all(m["support"][n] == 0 for n in CLASS_LABELS)

    def test_per_class_has_all_seven_keys(self) -> None:
        y = np.eye(NUM_CLASSES, dtype=np.float32)
        m = compute_metrics(y, y)
        assert set(m["per_class"].keys()) == set(CLASS_LABELS)
        for name in CLASS_LABELS:
            sub = m["per_class"][name]
            assert set(sub.keys()) == {"precision", "recall", "f1"}

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_metrics(
                np.zeros((3, NUM_CLASSES), dtype=np.float32),
                np.zeros((2, NUM_CLASSES), dtype=np.float32),
            )

    def test_wrong_columns_raises(self) -> None:
        bad = np.zeros((3, NUM_CLASSES + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="expected"):
            compute_metrics(bad, bad)


class TestSaveChart:
    def test_writes_png(self, tmp_path: Path) -> None:
        per_class = {
            n: {"precision": 0.5, "recall": 0.5, "f1": 0.5} for n in CLASS_LABELS
        }
        out = save_per_class_chart(per_class, tmp_path)
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 0

    def test_missing_class_raises(self, tmp_path: Path) -> None:
        partial = {n: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for n in list(CLASS_LABELS)[:-1]}
        with pytest.raises(ValueError, match="missing class"):
            save_per_class_chart(partial, tmp_path)


class TestRunEvaluation:
    def _probs_perfect(self) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.eye(NUM_CLASSES, dtype=np.float32)
        # Probabilities >= 0.5 only at the diagonal.
        probs = (y_true * 0.9) + ((1 - y_true) * 0.1)
        return probs.astype(np.float32), y_true

    def test_metrics_json_written(self, tmp_path: Path) -> None:
        probs, y_true = self._probs_perfect()
        run_evaluation(
            probs=probs,
            y_true=y_true,
            threshold=0.5,
            output_dir=tmp_path,
        )
        metrics_path = tmp_path / _METRICS_FILENAME
        assert metrics_path.is_file()
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in (
            "threshold",
            "accuracy",
            "macro_f1",
            "sample_f1",
            "hamming_loss",
            "per_class",
            "class_order",
            "chart_png",
            "support",
        ):
            assert key in payload, f"missing {key}"

    def test_threshold_value_in_metrics(self, tmp_path: Path) -> None:
        probs, y_true = self._probs_perfect()
        run_evaluation(
            probs=probs, y_true=y_true, threshold=0.42, output_dir=tmp_path
        )
        payload = json.loads((tmp_path / _METRICS_FILENAME).read_text(encoding="utf-8"))
        assert payload["threshold"] == 0.42

    def test_chart_png_saved(self, tmp_path: Path) -> None:
        probs, y_true = self._probs_perfect()
        run_evaluation(
            probs=probs, y_true=y_true, threshold=0.5, output_dir=tmp_path
        )
        chart = tmp_path / _CHART_FILENAME
        assert chart.is_file()
        assert chart.stat().st_size > 0

    def test_per_class_seven_keys_match_labels(self, tmp_path: Path) -> None:
        probs, y_true = self._probs_perfect()
        run_evaluation(
            probs=probs, y_true=y_true, threshold=0.5, output_dir=tmp_path
        )
        payload = json.loads((tmp_path / _METRICS_FILENAME).read_text(encoding="utf-8"))
        assert len(payload["per_class"]) == NUM_CLASSES
        assert set(payload["per_class"].keys()) == set(CLASS_LABELS)

    def test_threshold_zero_predicts_all_classes(self, tmp_path: Path) -> None:
        probs = np.full((3, NUM_CLASSES), 0.05, dtype=np.float32)
        y_true = np.zeros((3, NUM_CLASSES), dtype=np.float32)
        run_evaluation(
            probs=probs, y_true=y_true, threshold=0.0, output_dir=tmp_path
        )
        payload = json.loads((tmp_path / _METRICS_FILENAME).read_text(encoding="utf-8"))
        # Hamming should be 1.0 — every cell predicted 1, every cell true 0.
        assert payload["hamming_loss"] == 1.0

    def test_threshold_one_predicts_no_classes(self, tmp_path: Path) -> None:
        probs = np.full((3, NUM_CLASSES), 0.95, dtype=np.float32)
        y_true = np.ones((3, NUM_CLASSES), dtype=np.float32)
        run_evaluation(
            probs=probs, y_true=y_true, threshold=1.0, output_dir=tmp_path
        )
        payload = json.loads((tmp_path / _METRICS_FILENAME).read_text(encoding="utf-8"))
        # Every cell predicted 0; ground truth all 1 → Hamming 1.0.
        assert payload["hamming_loss"] == 1.0

    def test_extra_metadata_merged(self, tmp_path: Path) -> None:
        probs, y_true = self._probs_perfect()
        run_evaluation(
            probs=probs,
            y_true=y_true,
            threshold=0.5,
            output_dir=tmp_path,
            extra_metadata={"checkpoint_path": "/foo/bar.pt"},
        )
        payload = json.loads((tmp_path / _METRICS_FILENAME).read_text(encoding="utf-8"))
        assert payload["checkpoint_path"] == "/foo/bar.pt"


class TestResolveThreshold:
    def test_cli_override_wins(self) -> None:
        assert _resolve_threshold(0.7, {"training": {"threshold": 0.3}}) == 0.7

    def test_falls_back_to_config(self) -> None:
        assert _resolve_threshold(None, {"training": {"threshold": 0.3}}) == 0.3

    def test_falls_back_to_default(self) -> None:
        assert _resolve_threshold(None, {}) == 0.5

    def test_non_dict_config_uses_default(self) -> None:
        assert _resolve_threshold(None, []) == 0.5  # type: ignore[arg-type]

    def test_invalid_cli_raises(self) -> None:
        with pytest.raises(ValueError):
            _resolve_threshold(2.0, {})
