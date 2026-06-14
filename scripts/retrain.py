"""Scheduled HITL retraining.

Pulls the therapist-correction corpus (from S3 or the local feedback dir),
augments the training manifest with it, retrains from the current production
checkpoint, evaluates on the held-out test set, and **promotes the new model
only if it beats the current one**.

Designed to be run by cron (see ``ops/cron/stuttering-ai-retrain.cron``). It is
idempotent and cost-aware: it skips the (expensive) retrain unless at least
``--min-new`` corrections have accumulated since the last successful run, which
it tracks in a small JSON state file.

Single-label: each correction contributes one row ``(audio_path, label)`` where
``label`` is the first of the therapist's ``correct_labels`` (the primary type).

This module's data layer (corpus loading + manifest augmentation) is unit
tested; the train/evaluate/promote steps shell out to the existing
``ai/training/train_with_init.py`` and ``ai/evaluation/evaluate.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.labels import CLASS_LABELS  # noqa: E402

logger = logging.getLogger("retrain")

_VALID = frozenset(str(c) for c in CLASS_LABELS)


@dataclass
class Correction:
    """One therapist correction staged for training."""

    id: str
    label: str  # primary corrected label (single-label)
    audio_bytes: bytes


# --------------------------------------------------------------------------- #
# Corpus loading (S3 or local) — the testable data layer.
# --------------------------------------------------------------------------- #
def load_corpus_s3(s3, prefix: str) -> list[Correction]:
    """Load corrections from S3: one ``<prefix><id>.json`` + ``<id>.wav`` each."""
    out: list[Correction] = []
    for key in s3.list_keys(prefix):
        if not key.endswith(".json"):
            continue
        meta = s3.get_json(key)
        corr = _correction_from_meta(meta, lambda k: s3.download_bytes(k))
        if corr is not None:
            out.append(corr)
    return out


def load_corpus_local(feedback_dir: str) -> list[Correction]:
    """Load corrections from the local ``feedback.jsonl`` manifest + audio dir."""
    base = Path(feedback_dir)
    manifest = base / "feedback.jsonl"
    if not manifest.exists():
        return []
    out: list[Correction] = []
    with manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            corr = _correction_from_meta(meta, lambda rel: (base / rel).read_bytes())
            if corr is not None:
                out.append(corr)
    return out


def _correction_from_meta(meta: dict, read_audio) -> Optional[Correction]:
    """Build a Correction from a metadata record, fetching its audio. Skips
    (with a warning) records whose label is missing/unknown or whose audio
    can't be read — one bad record must not abort the whole run."""
    labels = meta.get("correct_labels") or []
    label = str(labels[0]) if labels else ""
    if label not in _VALID:
        logger.warning("Skipping correction %s: invalid label %r", meta.get("id"), label)
        return None
    audio_ref = meta.get("audio_file")
    if not audio_ref:
        logger.warning("Skipping correction %s: no audio_file", meta.get("id"))
        return None
    try:
        audio_bytes = read_audio(audio_ref)
    except Exception as exc:  # noqa: BLE001 — one unreadable clip mustn't abort
        logger.warning("Skipping correction %s: audio unreadable (%s)", meta.get("id"), exc)
        return None
    return Correction(id=str(meta.get("id", "")), label=label, audio_bytes=audio_bytes)


def stage_corpus(corrections: list[Correction], staging_dir: Path) -> list[tuple[str, str]]:
    """Write each correction's audio under ``staging_dir`` and return
    ``(absolute_audio_path, label)`` rows for the manifest."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    for corr in corrections:
        path = staging_dir / f"{corr.id}.wav"
        path.write_bytes(corr.audio_bytes)
        rows.append((str(path.resolve()), corr.label))
    return rows


def build_augmented_manifest(
    base_manifest: Path, correction_rows: list[tuple[str, str]], out_path: Path
) -> int:
    """Write ``out_path`` = the base train manifest + the correction rows.
    Returns the total row count. Assumes a header of ``audio_path,label`` (the
    repo manifest schema); extra base columns are preserved as-is."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with base_manifest.open("r", encoding="utf-8", newline="") as fin:
        reader = list(csv.reader(fin))
    header = reader[0]
    try:
        path_i, label_i = header.index("audio_path"), header.index("label")
    except ValueError as exc:
        raise ValueError(
            f"base manifest {base_manifest} must have 'audio_path' and 'label' columns"
        ) from exc
    rows = reader[1:]
    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        writer.writerows(rows)
        for audio_path, label in correction_rows:
            new_row = [""] * len(header)
            new_row[path_i] = audio_path
            new_row[label_i] = label
            writer.writerow(new_row)
    return len(rows) + len(correction_rows)


# --------------------------------------------------------------------------- #
# State (so cron only retrains on enough new data).
# --------------------------------------------------------------------------- #
def read_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"last_trained_count": 0, "last_macro_f1": None}


def write_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration (shells out to the existing train/eval scripts).
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], env: Optional[dict] = None) -> None:
    logger.info("run: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT), env=env)


def read_metric(metrics_json: Path, key: str = "f1_macro") -> Optional[float]:
    if not metrics_json.exists():
        return None
    data = json.loads(metrics_json.read_text(encoding="utf-8"))
    val = data.get(key)
    return float(val) if val is not None else None


def derive_training_config(
    base_config: Path, augmented_manifest: Path, work: Path
) -> tuple[Path, Path]:
    """Write a derived YAML config that points training at the augmented manifest
    and a per-run checkpoint dir. Returns (derived_config_path, candidate_ckpt).

    Without this, train.py would read the STATIC train manifest + checkpoint dir
    from the base config — the corrections would never reach training.
    """
    import yaml

    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    cfg.setdefault("data", {})["train_manifest"] = str(augmented_manifest)
    out = cfg.setdefault("output", {})
    out["checkpoint_dir"] = str(work / "checkpoints")
    out["experiment_name"] = "retrain"
    derived = work / "derived_config.yaml"
    derived.parent.mkdir(parents=True, exist_ok=True)
    derived.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    # train.py writes best_model.pt under <checkpoint_dir>/<experiment_name>/.
    candidate = work / "checkpoints" / "retrain" / "best_model.pt"
    return derived, candidate


def load_corpus_from_settings(settings) -> list[Correction]:
    """Load the correction corpus from whichever backend is configured."""
    if settings.STORAGE_BACKEND == "s3":
        from backend.services.s3_storage import S3Storage

        s3 = S3Storage(
            settings.S3_BUCKET,
            region=settings.S3_REGION,
            endpoint_url=settings.S3_ENDPOINT_URL,
        )
        return load_corpus_s3(s3, settings.S3_FEEDBACK_PREFIX)
    return load_corpus_local(settings.FEEDBACK_DIR)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scheduled HITL retraining.")
    parser.add_argument("--config", required=True, help="training YAML config")
    parser.add_argument("--base-manifest", required=True, help="base train manifest CSV")
    parser.add_argument("--test-manifest", required=True, help="held-out test manifest CSV")
    parser.add_argument("--init-checkpoint", required=True, help="current production .pt to fine-tune from")
    parser.add_argument("--work-dir", default="runs/retrain", help="scratch + output dir")
    parser.add_argument("--state-file", default="runs/retrain/state.json")
    parser.add_argument("--min-new", type=int, default=200, help="min new corrections before retraining")
    parser.add_argument("--dry-run", action="store_true", help="prepare data but don't train/promote")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    work = Path(args.work_dir)
    state_path = Path(args.state_file)
    state = read_state(state_path)

    from backend.app.config import get_settings

    corpus = load_corpus_from_settings(get_settings())
    # max(0, ...) so a corpus that shrank (S3 object expiry) can't go negative
    # and permanently stall retraining. The HITL corpus is append-only by design.
    new_count = max(0, len(corpus) - int(state.get("last_trained_count", 0)))
    logger.info("corpus=%d, new since last train=%d (min-new=%d)", len(corpus), new_count, args.min_new)
    if new_count < args.min_new:
        logger.info("Not enough new corrections — skipping retrain.")
        return 0

    rows = stage_corpus(corpus, work / "staged_audio")
    manifest = work / "augmented_train_manifest.csv"
    total = build_augmented_manifest(Path(args.base_manifest), rows, manifest)
    logger.info("augmented manifest: %d rows -> %s", total, manifest)

    if args.dry_run:
        logger.info("dry-run: stopping before train/evaluate/promote.")
        return 0

    # Retrain from the current production checkpoint on the AUGMENTED manifest.
    # A derived config wires train.py to the augmented manifest + a per-run
    # checkpoint dir; INIT_CHECKPOINT_PATH makes train_with_init.py fine-tune
    # from the incumbent instead of training from scratch.
    derived_config, new_ckpt = derive_training_config(Path(args.config), manifest, work)
    train_env = {**os.environ, "INIT_CHECKPOINT_PATH": str(args.init_checkpoint)}
    _run(
        [sys.executable, "ai/training/train_with_init.py", "--config", str(derived_config)],
        env=train_env,
    )

    # Evaluate the candidate and compare against the incumbent.
    eval_dir = work / "eval"
    _run([
        sys.executable, "ai/evaluation/evaluate.py",
        "--checkpoint_path", str(new_ckpt),
        "--manifest_path", args.test_manifest,
        "--output_dir", str(eval_dir),
    ])
    new_f1 = read_metric(eval_dir / "metrics.json")
    prev_f1 = state.get("last_macro_f1")
    logger.info("candidate macro_f1=%s, incumbent=%s", new_f1, prev_f1)

    promoted = new_f1 is not None and (prev_f1 is None or new_f1 > prev_f1)
    if promoted:
        logger.info("PROMOTE: candidate beats incumbent — deploy %s", new_ckpt)
        # (Deployment of the new artifact — upload to S3 / swap MODEL_PATH — is
        #  environment-specific and intentionally left to the deploy step.)
        state["last_macro_f1"] = new_f1
    else:
        logger.info("KEEP incumbent: candidate did not improve macro_f1.")
    state["last_trained_count"] = len(corpus)
    write_state(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
