"""Multi-label test-set evaluation: load checkpoint, score, write metrics + chart.

CLI:
    python ai/evaluation/evaluate.py \\
        --checkpoint_path ai/training/checkpoints/exp/best_model.pt \\
        --manifest_path   ai/dataset/processed/test_manifest.csv \\
        --output_dir      ai/evaluation/runs/my_run \\
        --threshold       0.5

Decoding: ``predicted = (sigmoid(logits) >= threshold).float()``.
Metrics are independent per class — values do NOT sum to 1.0.

Outputs in ``output_dir``:
    metrics.json — all metrics, threshold, per-class precision/recall/f1
    per_class_metrics.png — grouped bar chart, one group per class
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — tests + headless servers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from transformers import Wav2Vec2Processor  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai.models.stuttering_classifier import ModelConfig, StutteringClassifier  # noqa: E402
from ai.training.checkpoint_utils import load_checkpoint  # noqa: E402
from ai.training.dataloader import get_dataloader  # noqa: E402
from shared.labels import CLASS_LABELS, NUM_CLASSES  # noqa: E402

_DEFAULT_THRESHOLD = 0.5
_CHART_FILENAME = "per_class_metrics.png"
_METRICS_FILENAME = "metrics.json"


def _validate_threshold(threshold: float) -> float:
    """Coerce + validate a decision threshold.

    Args:
        threshold: Candidate threshold value.

    Returns:
        The threshold as a float, after range validation.

    Raises:
        ValueError: If ``threshold`` is not in [0.0, 1.0].
    """
    t = float(threshold)
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"threshold must be in [0.0, 1.0]; got {t}")
    return t


def decode(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Convert per-class sigmoid probabilities to multi-hot predictions.

    Args:
        probs: Array of shape ``(N, NUM_CLASSES)`` with values in [0, 1].
        threshold: Decision threshold; class included when ``prob >= threshold``.

    Returns:
        ``np.ndarray`` of shape ``(N, NUM_CLASSES)``, dtype float32, values in
        ``{0.0, 1.0}``.

    Raises:
        ValueError: On wrong shape, wrong dtype-ish input, or bad threshold.
    """
    t = _validate_threshold(threshold)
    arr = np.asarray(probs)
    if arr.ndim != 2 or arr.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"probs must have shape (N, {NUM_CLASSES}); got {arr.shape}"
        )
    return (arr >= t).astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute multi-label metrics on aligned multi-hot arrays.

    Args:
        y_true: Ground-truth multi-hot array, shape ``(N, NUM_CLASSES)``.
        y_pred: Predicted multi-hot array, shape ``(N, NUM_CLASSES)``.

    Returns:
        Dict with keys: ``accuracy`` (exact match), ``macro_f1``,
        ``sample_f1``, ``hamming_loss``, ``per_class`` (dict of
        ``CLASS_LABELS`` → ``{precision, recall, f1}``), ``support``.

    Raises:
        ValueError: If shapes mismatch or are malformed.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if y_true.size == 0:
        zeros = {n: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for n in CLASS_LABELS}
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "sample_f1": 0.0,
            "hamming_loss": 0.0,
            "per_class": zeros,
            "support": {n: 0 for n in CLASS_LABELS},
        }
    if y_true.ndim != 2 or y_true.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"expected (N, {NUM_CLASSES}) arrays; got {y_true.shape}"
        )

    accuracy = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    samples = float(f1_score(y_true, y_pred, average="samples", zero_division=0))
    ham = float(hamming_loss(y_true, y_pred))

    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class: dict[str, dict[str, float]] = {}
    support: dict[str, int] = {}
    for i, name in enumerate(CLASS_LABELS):
        per_class[name] = {
            "precision": float(prec_arr[i]) if i < len(prec_arr) else 0.0,
            "recall": float(rec_arr[i]) if i < len(rec_arr) else 0.0,
            "f1": float(f1_arr[i]) if i < len(f1_arr) else 0.0,
        }
        support[name] = int(sup_arr[i]) if i < len(sup_arr) else 0
    return {
        "accuracy": accuracy,
        "macro_f1": macro,
        "sample_f1": samples,
        "hamming_loss": ham,
        "per_class": per_class,
        "support": support,
    }


def save_per_class_chart(
    per_class: dict[str, dict[str, float]], output_dir: Path
) -> Path:
    """Save a grouped bar chart with precision/recall/f1 per class.

    Args:
        per_class: Mapping from class name to ``{precision, recall, f1}``.
        output_dir: Directory in which the PNG is created (must already exist).

    Returns:
        Path to the saved PNG.

    Raises:
        ValueError: If a class is missing from ``per_class``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in CLASS_LABELS:
        if name not in per_class:
            raise ValueError(f"per_class missing class {name!r}")
    precisions = [per_class[n]["precision"] for n in CLASS_LABELS]
    recalls = [per_class[n]["recall"] for n in CLASS_LABELS]
    f1s = [per_class[n]["f1"] for n in CLASS_LABELS]

    x = np.arange(NUM_CLASSES)
    bar_width = 0.27
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - bar_width, precisions, bar_width, label="precision")
    ax.bar(x, recalls, bar_width, label="recall")
    ax.bar(x + bar_width, f1s, bar_width, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(list(CLASS_LABELS), rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Per-class precision / recall / F1 (multi-label)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path = output_dir / _CHART_FILENAME
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_evaluation(
    *,
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    output_dir: Path,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode, score, and persist metrics + chart.

    Args:
        probs: Per-sample sigmoid probabilities, ``(N, NUM_CLASSES)``.
        y_true: Ground-truth multi-hot, ``(N, NUM_CLASSES)``.
        threshold: Decision threshold in [0, 1].
        output_dir: Directory for ``metrics.json`` and the PNG chart.
        extra_metadata: Optional fields merged into ``metrics.json`` (e.g.
            checkpoint_path, manifest_path) — useful for the CLI driver.

    Returns:
        The metrics payload that was written to disk.

    Raises:
        ValueError: For invalid threshold or shape mismatches.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t = _validate_threshold(threshold)
    y_pred = decode(probs, t)
    metrics = compute_metrics(np.asarray(y_true), y_pred)

    chart_path = save_per_class_chart(metrics["per_class"], output_dir)
    payload: dict[str, Any] = {
        "threshold": t,
        "class_order": list(CLASS_LABELS),
        "chart_png": str(chart_path),
        **metrics,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    metrics_path = output_dir / _METRICS_FILENAME
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def _get_device(device_cfg: str) -> torch.device:
    """Resolve compute device string ('auto', 'cpu', 'cuda', 'mps')."""
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(cfg: dict, device: torch.device) -> StutteringClassifier:
    """Instantiate the classifier from a saved checkpoint's config block."""
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


@torch.no_grad()
def _collect_probs_and_labels(
    model: StutteringClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a DataLoader and return (probs, y_true) arrays."""
    model.eval()
    probs_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).float()
        logits = model(input_values, attention_mask=attention_mask)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs_chunks.append(probs)
        labels_chunks.append(labels.detach().cpu().numpy())
    if not probs_chunks:
        empty = np.empty((0, NUM_CLASSES), dtype=np.float32)
        return empty, empty
    return (
        np.concatenate(probs_chunks, axis=0),
        np.concatenate(labels_chunks, axis=0),
    )


def _resolve_threshold(
    cli_threshold: float | None, ckpt_cfg: dict
) -> float:
    """Pick the threshold to use: CLI override > checkpoint config > default."""
    if cli_threshold is not None:
        return _validate_threshold(cli_threshold)
    cfg_value = (
        ckpt_cfg.get("training", {}).get("threshold")
        if isinstance(ckpt_cfg, dict)
        else None
    )
    if cfg_value is not None:
        return _validate_threshold(cfg_value)
    return _DEFAULT_THRESHOLD


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate StutteringClassifier on a test manifest (multi-label)."
    )
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--manifest_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Override decision threshold (else read from checkpoint config, "
        f"falling back to {_DEFAULT_THRESHOLD}).",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_path).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    preview = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(preview, dict) or "config" not in preview:
        raise ValueError(f"Checkpoint missing 'config' dict: {ckpt_path}")
    cfg = preview["config"]
    model = _build_model(cfg, torch.device("cpu"))
    meta = load_checkpoint(str(ckpt_path), model, optimizer=None)
    cfg = meta["config"]

    threshold = _resolve_threshold(args.threshold, cfg)
    device = _get_device(str(cfg.get("output", {}).get("device", "auto")))
    model = model.to(device)

    training = cfg.get("training", {})
    batch_size = int(training.get("batch_size", 16))
    num_workers = int(training.get("num_workers", 0))
    processor = Wav2Vec2Processor.from_pretrained(str(cfg["model"]["model_name"]))
    loader = get_dataloader(
        manifest_path,
        processor,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    if len(loader) == 0:
        raise RuntimeError(f"DataLoader is empty (check manifest): {manifest_path}")

    probs, y_true = _collect_probs_and_labels(model, loader, device)
    payload = run_evaluation(
        probs=probs,
        y_true=y_true,
        threshold=threshold,
        output_dir=output_dir,
        extra_metadata={
            "checkpoint_path": str(ckpt_path),
            "manifest_path": str(manifest_path),
            "output_dir": str(output_dir),
        },
    )
    print(f"accuracy={payload['accuracy']:.4f} | macro_f1={payload['macro_f1']:.4f} | "
          f"sample_f1={payload['sample_f1']:.4f} | hamming={payload['hamming_loss']:.4f} | "
          f"threshold={payload['threshold']:.2f}")
    print(f"Wrote metrics: {output_dir / _METRICS_FILENAME}")
    print(f"Wrote chart: {output_dir / _CHART_FILENAME}")


if __name__ == "__main__":
    main()
