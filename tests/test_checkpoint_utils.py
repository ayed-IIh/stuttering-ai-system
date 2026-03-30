"""Unit tests for ai.training.checkpoint_utils."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from ai.training.checkpoint_utils import (
    REQUIRED_CHECKPOINT_KEYS,
    export_model_for_inference,
    load_checkpoint,
    save_checkpoint,
)


def _minimal_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    *,
    epoch: int = 1,
    best_val_loss: float = 0.5,
    best_val_f1: float = 0.75,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "config": {"model": {"num_classes": 7}, "training": {"seed": 42}},
        "label2id": {"fluent": 0, "blocks": 1},
    }


def test_save_checkpoint_requires_all_keys(tmp_path: Path):
    m = nn.Linear(3, 2)
    opt = optim.Adam(m.parameters(), lr=1e-3)
    path = tmp_path / "c.pt"
    bad = {k: v for k, v in _minimal_state(m, opt).items() if k != "label2id"}
    with pytest.raises(ValueError, match="missing required keys"):
        save_checkpoint(bad, str(path))


def test_save_checkpoint_validates_types(tmp_path: Path):
    m = nn.Linear(3, 2)
    opt = optim.Adam(m.parameters(), lr=1e-3)
    path = tmp_path / "c.pt"
    state = _minimal_state(m, opt)
    state["epoch"] = "1"  # type: ignore[assignment]
    with pytest.raises(TypeError, match="epoch must be int"):
        save_checkpoint(state, str(path))


def test_save_and_load_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(0)
    m = nn.Linear(4, 3)
    opt = optim.Adam(m.parameters(), lr=0.01)
    x = torch.randn(2, 4)
    loss = m(x).sum()
    loss.backward()
    opt.step()

    path = tmp_path / "ckpt.pt"
    state = _minimal_state(m, opt, epoch=3, best_val_loss=0.1, best_val_f1=0.9)
    state["extra"] = {"note": "test"}
    save_checkpoint(state, str(path))

    m2 = nn.Linear(4, 3)
    opt2 = optim.Adam(m2.parameters(), lr=0.01)
    meta = load_checkpoint(str(path), m2, optimizer=opt2)

    assert torch.allclose(m.weight, m2.weight)
    assert torch.allclose(m.bias, m2.bias)
    assert meta["epoch"] == 3
    assert meta["best_val_loss"] == pytest.approx(0.1)
    assert meta["best_val_f1"] == pytest.approx(0.9)
    assert meta["config"]["training"]["seed"] == 42
    assert meta["label2id"]["fluent"] == 0
    assert meta["extra"]["note"] == "test"

    # Optimizer state restored (same step counts / momentum buffers)
    assert opt2.state_dict()["state"]


def test_load_checkpoint_without_optimizer(tmp_path: Path):
    m = nn.Linear(2, 2)
    opt = optim.SGD(m.parameters(), lr=0.1)
    path = tmp_path / "c.pt"
    save_checkpoint(_minimal_state(m, opt), str(path))

    m2 = nn.Linear(2, 2)
    meta = load_checkpoint(str(path), m2, optimizer=None)
    assert "optimizer_state_dict" not in meta
    assert torch.allclose(m.weight, m2.weight)


def test_export_model_for_inference_writes_files(tmp_path: Path):
    m = nn.Linear(5, 2)
    opt = optim.Adam(m.parameters())
    ckpt_path = tmp_path / "train.pt"
    save_checkpoint(_minimal_state(m, opt), str(ckpt_path))

    out_dir = tmp_path / "export"
    export_model_for_inference(str(ckpt_path), str(out_dir))

    inf_path = out_dir / "model_inference.pt"
    cfg_path = out_dir / "config.json"
    assert inf_path.is_file()
    assert cfg_path.is_file()

    loaded = torch.load(inf_path, map_location="cpu", weights_only=False)
    assert set(loaded.keys()) == {"model_state_dict", "config"}
    assert torch.allclose(loaded["model_state_dict"]["weight"], m.weight)

    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["model"]["num_classes"] == 7


def test_cli_export_module_invocation(tmp_path: Path):
    """``python -m ai.training.checkpoint_utils export`` writes inference artifacts."""
    m = nn.Linear(3, 2)
    opt = optim.Adam(m.parameters())
    ckpt = tmp_path / "full.pt"
    save_checkpoint(_minimal_state(m, opt), str(ckpt))

    out = tmp_path / "out_cli"
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai.training.checkpoint_utils",
            "export",
            "--checkpoint_path",
            str(ckpt),
            "--output_dir",
            str(out),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "model_inference.pt").is_file()
    assert (out / "config.json").is_file()


def test_required_keys_matches_contract():
    assert "model_state_dict" in REQUIRED_CHECKPOINT_KEYS
    assert "optimizer_state_dict" in REQUIRED_CHECKPOINT_KEYS
    assert "epoch" in REQUIRED_CHECKPOINT_KEYS
    assert "best_val_loss" in REQUIRED_CHECKPOINT_KEYS
    assert "best_val_f1" in REQUIRED_CHECKPOINT_KEYS
    assert "config" in REQUIRED_CHECKPOINT_KEYS
    assert "label2id" in REQUIRED_CHECKPOINT_KEYS
    assert len(REQUIRED_CHECKPOINT_KEYS) == 7
