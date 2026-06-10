"""
Standalone test-set evaluation: load a training checkpoint, run inference, report metrics.

Example:
  python ai/evaluation/evaluate.py \\
    --checkpoint_path ai/training/checkpoints/exp/best_model.pt \\
    --manifest_path ai/dataset/processed/test_manifest.csv \\
    --output_dir ai/evaluation/runs/my_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import Wav2Vec2Processor

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai.models.stuttering_classifier import ModelConfig, StutteringClassifier  # noqa: E402
from ai.training.checkpoint_utils import load_checkpoint  # noqa: E402
from ai.training.dataloader import get_dataloader  # noqa: E402
from shared.labels import CLASS_LABELS, NUM_CLASSES  # noqa: E402

_REPETITION_SUBCLASSES = frozenset(
    {"part_word_repetition", "phrase_repetition", "word_repetition"}
)
_REPETITION_RECALL_WARN_THRESHOLD = 0.50


def _get_device(device_cfg: str) -> torch.device:
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def _collect_predictions(
    model: StutteringClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_values, attention_mask=attention_mask)
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
            all_preds.extend(int(p) for p in preds)
            all_labels.extend(int(x) for x in labels.cpu().numpy().tolist())
    return all_labels, all_preds


def _build_data_warning(per_class_recall: dict[str, float]) -> dict | None:
    flagged: list[dict[str, float | str]] = []
    for name in CLASS_LABELS:
        if name not in _REPETITION_SUBCLASSES:
            continue
        r = per_class_recall.get(name, 0.0)
        if r < _REPETITION_RECALL_WARN_THRESHOLD:
            flagged.append({"class": name, "recall": float(r)})
    if not flagged:
        return None
    return {
        "kind": "repetition_subclass_recall_below_0_50",
        "threshold": _REPETITION_RECALL_WARN_THRESHOLD,
        "flagged_classes": flagged,
        "message": (
            "One or more repetition subclasses have recall below 0.50; "
            "this may reflect limited training data. See "
            "ai/evaluation/ACCEPTANCE_CRITERIA.md and align thresholds with the team."
        ),
    }


def _print_summary_table(
    accuracy: float,
    f1_macro: float,
    f1_weighted: float,
    per_class: dict[str, dict[str, float]],
    support: dict[str, int],
) -> None:
    w_class = max(len("class"), max(len(c) for c in CLASS_LABELS))
    print()
    print("=== Test set evaluation ===")
    print(f"{'Metric':<24} {'Value':>10}")
    print("-" * 36)
    print(f"{'Accuracy':<24} {accuracy:>10.4f}")
    print(f"{'F1 (macro)':<24} {f1_macro:>10.4f}")
    print(f"{'F1 (weighted)':<24} {f1_weighted:>10.4f}")
    print()
    hdr = (
        f"{'Class':<{w_class}}  {'Precision':>10}  {'Recall':>10}  "
        f"{'F1':>10}  {'Support':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name in CLASS_LABELS:
        m = per_class[name]
        n = support.get(name, 0)
        print(
            f"{name:<{w_class}}  {m['precision']:>10.4f}  {m['recall']:>10.4f}  "
            f"{m['f1']:>10.4f}  {n:>8d}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate StutteringClassifier on a test manifest (no gradients)."
    )
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--manifest_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_path).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Peek config so the model architecture matches weights; load_checkpoint loads again.
    preview = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(preview, dict) or "config" not in preview:
        raise ValueError(f"Checkpoint missing 'config' dict: {ckpt_path}")

    cfg = preview["config"]
    model = _build_model(cfg, torch.device("cpu"))
    meta = load_checkpoint(str(ckpt_path), model, optimizer=None)
    cfg = meta["config"]
    label2id = {str(k): int(v) for k, v in meta["label2id"].items()}

    device = _get_device(str(cfg.get("output", {}).get("device", "auto")))
    model = model.to(device)

    training = cfg.get("training", {})
    batch_size = int(training.get("batch_size", 16))
    num_workers = int(training.get("num_workers", 0))

    model_name = str(cfg["model"]["model_name"])
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
    except (OSError, EnvironmentError):
        # XLS-R and other encoder-only checkpoints ship without a tokenizer.
        # For classification we only need the feature extractor.
        from transformers import Wav2Vec2FeatureExtractor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    loader = get_dataloader(
        manifest_path,
        processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        label_to_id=label2id,
    )
    if len(loader) == 0:
        raise RuntimeError(f"DataLoader is empty (check manifest): {manifest_path}")

    y_true, y_pred = _collect_predictions(model, loader, device)
    labels_idx = list(range(NUM_CLASSES))

    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_idx,
        zero_division=0,
    )
    per_class: dict[str, dict[str, float]] = {}
    support: dict[str, int] = {}
    per_class_recall: dict[str, float] = {}
    for i, name in enumerate(CLASS_LABELS):
        per_class[name] = {
            "precision": float(prec_arr[i]),
            "recall": float(rec_arr[i]),
            "f1": float(f1_arr[i]),
        }
        support[name] = int(sup_arr[i])
        per_class_recall[name] = float(rec_arr[i])

    cm_counts = confusion_matrix(y_true, y_pred, labels=labels_idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_norm = cm_counts.astype(np.float64) / row_sums
    cm_norm = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(list(CLASS_LABELS), rotation=45, ha="right")
    ax.set_yticklabels(list(CLASS_LABELS))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix (row-normalized)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=8,
            )
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    data_warning = _build_data_warning(per_class_recall)

    metrics_payload: dict = {
        "checkpoint_path": str(ckpt_path),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class": per_class,
        "support": support,
        "confusion_matrix_normalized": cm_norm.tolist(),
        "confusion_matrix_counts": cm_counts.tolist(),
        "class_order": list(CLASS_LABELS),
        "confusion_matrix_png": str(cm_path),
        "data_warning": data_warning,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    _print_summary_table(accuracy, f1_macro, f1_weighted, per_class, support)
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote confusion matrix: {cm_path}")
    if data_warning:
        print("data_warning:", json.dumps(data_warning, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
