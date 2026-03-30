"""Checkpoint persistence, training log helpers, and inference export (ADN-04)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

REQUIRED_CHECKPOINT_KEYS: frozenset[str] = frozenset(
    {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "best_val_loss",
        "best_val_f1",
        "config",
        "label2id",
    }
)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_checkpoint_state(state: Mapping[str, Any]) -> None:
    missing = REQUIRED_CHECKPOINT_KEYS - state.keys()
    if missing:
        raise ValueError(
            f"Checkpoint state missing required keys: {sorted(missing)}. "
            f"Required: {sorted(REQUIRED_CHECKPOINT_KEYS)}."
        )
    if not isinstance(state["epoch"], int):
        raise TypeError(f"epoch must be int, got {type(state['epoch']).__name__}")
    if not isinstance(state["config"], dict):
        raise TypeError(f"config must be dict, got {type(state['config']).__name__}")
    if not isinstance(state["label2id"], dict):
        raise TypeError(f"label2id must be dict, got {type(state['label2id']).__name__}")


def save_checkpoint(state: dict[str, Any], filepath: str) -> None:
    """Persist a training checkpoint via ``torch.save``.

    ``state`` must include: ``model_state_dict``, ``optimizer_state_dict``, ``epoch``,
    ``best_val_loss``, ``best_val_f1``, ``config``, ``label2id``. Additional keys
    (e.g. ``scheduler_state_dict``, ``metrics``) are allowed.
    """
    _validate_checkpoint_state(state)
    path = Path(filepath)
    ensure_parent_dir(path)
    torch.save(state, path)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load checkpoint, restore ``model`` and optionally ``optimizer``, return metadata.

    Returns a dict of all stored keys except ``model_state_dict`` and
    ``optimizer_state_dict`` (e.g. ``epoch``, ``best_val_loss``, ``best_val_f1``,
    ``config``, ``label2id``, ``scheduler_state_dict``, ``metrics``, ``extra``).
    """
    path = Path(filepath)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt).__name__}")
    _validate_checkpoint_state(ckpt)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return {
        k: v
        for k, v in ckpt.items()
        if k not in ("model_state_dict", "optimizer_state_dict")
    }


def export_model_for_inference(checkpoint_path: str, output_dir: str) -> None:
    """Load a training checkpoint and write inference-only artifacts for backend handoff.

    Writes ``output_dir/model_inference.pt`` (``model_state_dict`` + ``config`` only) and
    ``output_dir/config.json`` (pretty-printed ``config``).
    """
    ckpt_path = Path(checkpoint_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt).__name__}")
    _validate_checkpoint_state(ckpt)

    inference_payload = {
        "model_state_dict": ckpt["model_state_dict"],
        "config": ckpt["config"],
    }
    torch.save(inference_payload, out / "model_inference.pt")

    config_path = out / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(ckpt["config"], f, indent=2, ensure_ascii=False, default=str)


def save_best_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer,
    val_macro_f1: float,
    val_loss: float,
    best_val_loss: float,
    best_val_f1: float,
    epoch: int,
    config: Mapping[str, Any],
    label2id: Mapping[str, int],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Save best model snapshot (full standard checkpoint) ranked by validation macro-F1."""
    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_f1": float(best_val_f1),
        "config": dict(config),
        "label2id": dict(label2id),
        "metrics": {
            "val_macro_f1": float(val_macro_f1),
            "val_loss": float(val_loss),
        },
    }
    if extra:
        state["extra"] = dict(extra)
    save_checkpoint(state, str(path))


def save_last_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    metrics: Mapping[str, Any],
    best_val_loss: float,
    best_val_f1: float,
    config: Mapping[str, Any],
    label2id: Mapping[str, int],
) -> None:
    """Save full training state for resume and end-of-epoch snapshots."""
    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_f1": float(best_val_f1),
        "config": dict(config),
        "label2id": dict(label2id),
        "metrics": dict(metrics),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    save_checkpoint(state, str(path))


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


def _cli_export(args: argparse.Namespace) -> None:
    export_model_for_inference(args.checkpoint_path, args.output_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Checkpoint utilities (export inference bundle).")
    sub = p.add_subparsers(dest="command", required=True)
    p_exp = sub.add_parser("export", help="Write model_inference.pt and config.json for inference")
    p_exp.add_argument("--checkpoint_path", required=True, help="Path to training checkpoint (.pt)")
    p_exp.add_argument("--output_dir", required=True, help="Directory for exported artifacts")
    p_exp.set_defaults(func=_cli_export)
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
