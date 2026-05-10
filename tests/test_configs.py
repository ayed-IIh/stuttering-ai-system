"""Tests for the training YAML configs — multi-label fields must be present."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "ai" / "training" / "configs"
)
_CONFIG_FILES = ("baseline_frozen.yaml", "finetune_full.yaml")


@pytest.fixture(params=_CONFIG_FILES)
def loaded_config(request) -> dict:
    path = _CONFIG_DIR / request.param
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_file_exists() -> None:
    for name in _CONFIG_FILES:
        assert (_CONFIG_DIR / name).is_file()


def test_loads_without_error(loaded_config: dict) -> None:
    assert isinstance(loaded_config, dict)
    for required_section in ("model", "training", "data", "output"):
        assert required_section in loaded_config


def test_contains_loss_field(loaded_config: dict) -> None:
    assert loaded_config["training"]["loss"] == "bce_with_logits"


def test_contains_threshold_field(loaded_config: dict) -> None:
    assert "threshold" in loaded_config["training"]


def test_threshold_in_unit_interval(loaded_config: dict) -> None:
    threshold = float(loaded_config["training"]["threshold"])
    assert 0.0 <= threshold <= 1.0
