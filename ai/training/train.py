"""
Main training entrypoint.

Example:
  python ai/training/train.py --config ai/training/configs/baseline_frozen.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from transformers import Wav2Vec2Processor, get_linear_schedule_with_warmup

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
from shared.labels import LABEL2ID  # noqa: E402


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_path(cwd: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (cwd / path).resolve()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(cfg: dict, device: torch.device) -> StutteringClassifier:
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


def _train_one_epoch(
    model: StutteringClassifier,
    loader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_values, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(model: StutteringClassifier, loader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_values, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, labels)

        total_loss += float(loss.item())
        n_batches += 1
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().tolist())

    mean_loss = total_loss / max(n_batches, 1)
    acc = float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0
    macro_f1 = float(
        f1_score(all_labels, all_preds, average="macro", zero_division=0)
    ) if all_labels else 0.0
    return mean_loss, acc, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StutteringClassifier (Wav2Vec2).")
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

    experiment_name = str(raw["experiment_name"])
    paths = raw["paths"]
    training = raw["training"]

    train_manifest = _resolve_path(cwd, paths["train_manifest"])
    val_manifest = _resolve_path(cwd, paths["val_manifest"])
    checkpoint_root = _resolve_path(cwd, paths["checkpoint_root"])
    log_root = _resolve_path(cwd, paths["log_root"])

    exp_ckpt_dir = checkpoint_root / experiment_name
    exp_log_dir = log_root / experiment_name
    best_model_path = exp_ckpt_dir / "best_model.pt"
    last_model_path = exp_ckpt_dir / "last_model.pt"
    training_log_csv = exp_log_dir / "training_log.csv"

    seed = int(training["seed"])
    _set_seed(seed)

    device = _get_device()

    model_cfg = raw["model"]
    processor = Wav2Vec2Processor.from_pretrained(model_cfg["model_name"])
    model = _build_model(raw, device)

    batch_size = int(training["batch_size"])
    num_workers = int(training["num_workers"])
    num_epochs = int(training["num_epochs"])
    lr = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    warmup_ratio = float(training["warmup_ratio"])
    max_grad_norm = float(training["max_grad_norm"])

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
        raise RuntimeError(f"Train DataLoader is empty (check manifest): {train_manifest}")
    if len(val_loader) == 0:
        raise RuntimeError(f"Validation DataLoader is empty (check manifest): {val_manifest}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)

    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    log_fields = (
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "learning_rate",
    )

    best_f1 = -1.0
    epoch = 0

    try:
        for epoch in range(1, num_epochs + 1):
            train_loss = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                max_grad_norm,
            )
            val_loss, val_acc, val_macro_f1 = _validate(model, val_loader, device)

            lr_now = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else lr

            append_training_log_csv(
                training_log_csv,
                log_fields,
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_accuracy": f"{val_acc:.6f}",
                    "val_macro_f1": f"{val_macro_f1:.6f}",
                    "learning_rate": f"{lr_now:.8e}",
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
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_macro_f1,
                },
            )

            if val_macro_f1 > best_f1:
                best_f1 = val_macro_f1
                save_best_checkpoint(
                    best_model_path,
                    model,
                    val_macro_f1=val_macro_f1,
                    epoch=epoch,
                    extra={"config_path": str(config_path)},
                )

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_accuracy={val_acc:.4f} | "
                f"val_macro_f1={val_macro_f1:.4f} | "
                f"lr={lr_now:.2e} | "
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
        )
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
