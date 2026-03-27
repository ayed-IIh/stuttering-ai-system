"""Checkpoint persistence and training log helpers (ADN-04)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_best_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    val_macro_f1: float,
    epoch: int,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Save best model weights ranked by validation macro-F1."""
    ensure_parent_dir(path)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "val_macro_f1": float(val_macro_f1),
        "epoch": int(epoch),
    }
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, path)


def save_last_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    metrics: Mapping[str, Any],
) -> None:
    """Save full training state for resume and end-of-epoch snapshots."""
    ensure_parent_dir(path)
    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def append_training_log_csv(
    csv_path: Path,
    fieldnames: Sequence[str],
    row: Mapping[str, Any],
) -> None:
    """Append one row to per-epoch metrics CSV (writes header on first create)."""
    ensure_parent_dir(csv_path)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
