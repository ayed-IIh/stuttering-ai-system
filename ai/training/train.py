"""Multi-label training entrypoint for the stuttering classifier.

Loss : ``F.binary_cross_entropy_with_logits`` (one independent binary head
       per class).
Decode: ``sigmoid(logits) >= threshold`` → multi-hot prediction.
Metrics: macro-F1, sample-F1, Hamming loss, plus per-epoch accuracy
       (exact-match across all 7 classes).

Threshold is read from ``config['training']['threshold']`` — never hardcoded.

Example:
    python ai/training/train.py --config ai/training/configs/baseline_frozen.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
)
from torch.optim import AdamW
from transformers import (
    Wav2Vec2Processor,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# Repo root on sys.path when running as `python ai/training/train.py`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai.models.stuttering_classifier import ModelConfig, StutteringClassifier  # noqa: E402
from ai.training.checkpoint_utils import (  # noqa: E402
    append_training_log_csv,
    save_best_checkpoint,
    save_last_checkpoint,
)
from ai.training.dataloader import get_dataloader  # noqa: E402
from shared.labels import LABEL2ID, NUM_CLASSES  # noqa: E402


def _get_device(device_cfg: str = "auto") -> torch.device:
    """Resolve compute device from a config string."""
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_path(cwd: Path, p: str | Path) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (cwd / path).resolve()


def _load_config(path: Path) -> dict:
    """Read a YAML config file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    """Seed all RNG sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(cfg: dict, device: torch.device) -> StutteringClassifier:
    """Instantiate StutteringClassifier from a config dict and move to device."""
    m = cfg["model"]
    mc = ModelConfig(
        model_name=m["model_name"],
        num_classes=int(m["num_classes"]),
        dropout_rate=float(m["dropout_rate"]),
        freeze_encoder=bool(m["freeze_encoder"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
    )
    model = StutteringClassifier(mc)
    return model.to(device)


def _decode_multi_hot(
    logits: torch.Tensor, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw logits to (multi-hot predictions, sigmoid probabilities).

    Args:
        logits: ``(batch, NUM_CLASSES)`` raw model output.
        threshold: Decision threshold in [0, 1].

    Returns:
        (predictions, probabilities) as numpy arrays of shape
        ``(batch, NUM_CLASSES)``. Predictions are float32 with values in
        ``{0.0, 1.0}``; probabilities are the post-sigmoid values.
    """
    if logits.dim() != 2 or logits.shape[1] != NUM_CLASSES:
        raise RuntimeError(
            f"Bad logits shape {tuple(logits.shape)}; expected (batch, {NUM_CLASSES})"
        )
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return (
        preds.detach().cpu().numpy(),
        probs.detach().cpu().numpy(),
    )


def _train_one_epoch(
    model: StutteringClassifier,
    loader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """Run one training epoch and return mean loss across batches."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_values, attention_mask=attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _validation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute multi-label validation metrics.

    Returns a dict with: exact_match_accuracy, macro_f1, sample_f1, hamming_loss.
    """
    if y_true.size == 0:
        return {
            "exact_match_accuracy": 0.0,
            "macro_f1": 0.0,
            "sample_f1": 0.0,
            "hamming_loss": 0.0,
        }
    return {
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "sample_f1": float(
            f1_score(y_true, y_pred, average="samples", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


@torch.no_grad()
def _validate(
    model: StutteringClassifier,
    loader,
    device: torch.device,
    threshold: float,
) -> tuple[float, dict[str, float]]:
    """Run validation and return (mean_loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).float()

        logits = model(input_values, attention_mask=attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss += float(loss.item())
        n_batches += 1

        preds, _ = _decode_multi_hot(logits, threshold)
        all_preds.append(preds)
        all_labels.append(labels.detach().cpu().numpy())

    mean_loss = total_loss / max(n_batches, 1)
    if all_preds:
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
    else:
        y_pred = np.empty((0, NUM_CLASSES))
        y_true = np.empty((0, NUM_CLASSES))
    metrics = _validation_metrics(y_true, y_pred)
    return mean_loss, metrics


def _read_threshold(training_cfg: dict) -> float:
    """Read and validate the multi-label decision threshold from config."""
    if "threshold" not in training_cfg:
        raise KeyError(
            "Missing required field 'threshold' in training config "
            "(multi-label decision threshold, e.g. 0.5)"
        )
    threshold = float(training_cfg["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"threshold must be in [0.0, 1.0]; got {threshold}"
        )
    return threshold


_VALID_SCHEDULER_TYPES = ("linear", "cosine")


def _build_scheduler(scheduler_type: str, optimizer, num_warmup_steps, num_training_steps):
    """Build the LR scheduler named by ``scheduler_type``.

    Raises:
        ValueError: If ``scheduler_type`` is not in ``_VALID_SCHEDULER_TYPES``.
            Falling back silently to ``linear`` would mask config typos and
            make experiment runs hard to reproduce.
    """
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    raise ValueError(
        f"Unknown scheduler_type {scheduler_type!r}; "
        f"valid options are {list(_VALID_SCHEDULER_TYPES)}"
    )


_LOG_FIELDS: tuple[str, ...] = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_exact_match_accuracy",
    "val_macro_f1",
    "val_sample_f1",
    "val_hamming_loss",
    "learning_rate",
    "threshold",
)


def run_training(
    raw: dict[str, Any],
    *,
    cwd: Path | None = None,
    config_path: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run training from an in-memory config dict.

    Returns best validation macro-F1 observed, validation loss at that epoch,
    and the experiment name used for checkpoints/logs.
    """
    if cwd is None:
        cwd = Path.cwd().resolve()

    data_cfg = raw["data"]
    output_cfg = raw["output"]
    training = raw["training"]

    threshold = _read_threshold(training)

    experiment_name = str(output_cfg["experiment_name"])
    train_manifest = _resolve_path(cwd, data_cfg["train_manifest"])
    val_manifest = _resolve_path(cwd, data_cfg["val_manifest"])
    checkpoint_root = _resolve_path(
        cwd, output_cfg.get("checkpoint_dir", "ai/training/checkpoints")
    )
    log_root = _resolve_path(cwd, output_cfg.get("log_dir", "ai/training/logs"))

    exp_ckpt_dir = checkpoint_root / experiment_name
    exp_log_dir = log_root / experiment_name
    best_model_path = exp_ckpt_dir / "best_model.pt"
    last_model_path = exp_ckpt_dir / "last_model.pt"
    training_log_csv = exp_log_dir / "training_log.csv"

    seed = int(training.get("seed", 42))
    _set_seed(seed)

    device_cfg = str(output_cfg.get("device", "auto"))
    device = _get_device(device_cfg)

    model_cfg = raw["model"]
    processor = Wav2Vec2Processor.from_pretrained(model_cfg["model_name"])
    model = _build_model(raw, device)

    batch_size = int(training["batch_size"])
    num_workers = int(training.get("num_workers", 0))
    num_epochs = int(training["num_epochs"])
    lr = float(training["learning_rate"])
    weight_decay = float(training.get("weight_decay", 0.01))
    warmup_ratio = float(training.get("warmup_ratio", 0.0))
    max_grad_norm = float(training.get("gradient_clip_norm", 1.0))
    scheduler_type = str(training.get("scheduler_type", "linear"))

    train_loader = get_dataloader(
        train_manifest,
        processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        label_to_id=LABEL2ID,
    )
    val_loader = get_dataloader(
        val_manifest,
        processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        label_to_id=LABEL2ID,
    )

    if len(train_loader) == 0:
        raise RuntimeError(f"Train DataLoader is empty: {train_manifest}")
    if len(val_loader) == 0:
        raise RuntimeError(f"Validation DataLoader is empty: {val_manifest}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = _build_scheduler(
        scheduler_type, optimizer, num_warmup_steps, num_training_steps
    )

    best_f1 = -1.0
    best_val_loss = float("inf")
    best_val_loss_at_best_f1 = float("inf")
    epoch = 0
    extra_cfg = {"config_path": str(config_path)} if config_path is not None else {}

    try:
        for epoch in range(1, num_epochs + 1):
            train_loss = _train_one_epoch(
                model, train_loader, optimizer, scheduler, device, max_grad_norm
            )
            val_loss, metrics = _validate(model, val_loader, device, threshold)
            best_val_loss = min(best_val_loss, val_loss)
            running_best_f1 = max(best_f1, metrics["macro_f1"])

            lr_now = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else lr

            append_training_log_csv(
                training_log_csv,
                _LOG_FIELDS,
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_exact_match_accuracy": f"{metrics['exact_match_accuracy']:.6f}",
                    "val_macro_f1": f"{metrics['macro_f1']:.6f}",
                    "val_sample_f1": f"{metrics['sample_f1']:.6f}",
                    "val_hamming_loss": f"{metrics['hamming_loss']:.6f}",
                    "learning_rate": f"{lr_now:.8e}",
                    "threshold": f"{threshold:.4f}",
                },
            )

            save_last_checkpoint(
                last_model_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in metrics.items()},
                    "threshold": threshold,
                },
                best_val_loss=best_val_loss,
                best_val_f1=running_best_f1,
                config=raw,
                label2id=dict(LABEL2ID),
            )

            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                best_val_loss_at_best_f1 = val_loss
                save_best_checkpoint(
                    best_model_path,
                    model,
                    optimizer=optimizer,
                    val_macro_f1=metrics["macro_f1"],
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    best_val_f1=best_f1,
                    epoch=epoch,
                    config=raw,
                    label2id=dict(LABEL2ID),
                    extra=extra_cfg,
                )

            if verbose:
                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_acc={metrics['exact_match_accuracy']:.4f} | "
                    f"val_macro_f1={metrics['macro_f1']:.4f} | "
                    f"val_sample_f1={metrics['sample_f1']:.4f} | "
                    f"val_hamming={metrics['hamming_loss']:.4f} | "
                    f"lr={lr_now:.2e} | "
                    f"thr={threshold:.2f} | "
                    f"best_val_macro_f1={best_f1:.4f}"
                )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: saving checkpoint before exit...", flush=True)
        save_last_checkpoint(
            last_model_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics={"interrupted": True, "note": "keyboard_interrupt"},
            best_val_loss=best_val_loss,
            best_val_f1=best_f1,
            config=raw,
            label2id=dict(LABEL2ID),
        )
        raise SystemExit(130) from None

    return {
        "best_val_macro_f1": float(best_f1),
        "val_loss_at_best_f1": float(best_val_loss_at_best_f1),
        "experiment_name": experiment_name,
    }


def main() -> None:
    """CLI entrypoint: `python train.py --config <path.yaml>`."""
    parser = argparse.ArgumentParser(
        description="Train StutteringClassifier (Wav2Vec2, multi-label)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (paths resolved relative to current working directory).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cwd = Path.cwd().resolve()
    raw = _load_config(config_path)
    run_training(raw, cwd=cwd, config_path=config_path, verbose=True)


if __name__ == "__main__":
    main()
