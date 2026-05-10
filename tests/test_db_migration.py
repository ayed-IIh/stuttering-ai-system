"""Tests for backend/db/migrations/002_multi_label_predictions.sql.

These tests check the SQL file's structure and content — they do not actually
run the migration against a Postgres instance. The migration is exercised in
integration tests (separate suite, CI-only).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.labels import CLASS_LABELS

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MIGRATION = (
    _REPO_ROOT / "backend" / "db" / "migrations" / "002_multi_label_predictions.sql"
)


@pytest.fixture(scope="module")
def sql_text() -> str:
    return _MIGRATION.read_text(encoding="utf-8")


def test_migration_file_exists() -> None:
    assert _MIGRATION.is_file(), f"missing migration file: {_MIGRATION}"
    assert _MIGRATION.stat().st_size > 0


def test_has_begin_and_commit(sql_text: str) -> None:
    assert "BEGIN;" in sql_text
    assert "COMMIT;" in sql_text


def test_prediction_classes_table_definition_present(sql_text: str) -> None:
    assert "CREATE TABLE IF NOT EXISTS prediction_classes" in sql_text
    # Each required column must appear by name.
    for col in (
        "prediction_id",
        "class_label",
        "confidence",
        "created_at",
    ):
        assert col in sql_text, f"missing column reference: {col}"


def test_rollback_comment_present(sql_text: str) -> None:
    assert "ROLLBACK INSTRUCTIONS" in sql_text
    # The rollback block must reference re-adding the legacy column.
    assert "ADD COLUMN predicted_class" in sql_text


@pytest.mark.parametrize("class_name", list(CLASS_LABELS))
def test_all_seven_classes_in_check_constraint(sql_text: str, class_name: str) -> None:
    # Each class name must literally appear in the file (in the CHECK array).
    assert f"'{class_name}'" in sql_text


def test_chk_class_label_constraint_name_present(sql_text: str) -> None:
    assert "CONSTRAINT chk_class_label" in sql_text


def test_indexes_created(sql_text: str) -> None:
    assert "idx_prediction_classes_prediction_id" in sql_text
    assert "idx_prediction_classes_class_label" in sql_text


def test_drops_predicted_class_column(sql_text: str) -> None:
    assert "DROP COLUMN IF EXISTS predicted_class" in sql_text


def test_independent_sigmoid_comment_present(sql_text: str) -> None:
    """The migration must remind future readers that values don't sum to 1.0."""
    assert "do NOT sum to 1.0" in sql_text or "NOT sum to 1.0" in sql_text


def test_foreign_key_with_cascade_delete(sql_text: str) -> None:
    assert "REFERENCES predictions(id) ON DELETE CASCADE" in sql_text


def test_confidence_check_in_unit_interval(sql_text: str) -> None:
    assert "CHECK (confidence BETWEEN 0 AND 1)" in sql_text


def test_unique_prediction_class_pair(sql_text: str) -> None:
    """Same class must not be insertable twice for the same prediction."""
    assert "uq_prediction_classes_prediction_id_class_label" in sql_text
    assert "UNIQUE (prediction_id, class_label)" in sql_text


def test_no_redundant_fk_constraint_name(sql_text: str) -> None:
    """Inline REFERENCES is the canonical FK; no separate named FK needed."""
    assert "CONSTRAINT fk_prediction" not in sql_text
