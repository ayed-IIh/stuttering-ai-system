#!/usr/bin/env python3
"""
Run pre-data readiness checks before the client dataset arrives.

This script focuses on:
1) smoke/integration quality checks
2) training pipeline sanity checks that do not require new client data
3) a machine-readable readiness report

Usage:
  python scripts/pre_data_readiness.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.labels import CLASS_LABELS, LABEL2ID, NUM_CLASSES

EXPECTED_LABELS: tuple[str, ...] = (
    "fluent",
    "blocks",
    "interjections",
    "prolongations",
    "part_word_repetition",
    "phrase_repetition",
    "word_repetition",
)

REQUIRED_FILES = (
    "ai/training/train.py",
    "ai/evaluation/evaluate.py",
    "ai/training/checkpoint_utils.py",
    "backend/app/main.py",
    "backend/services/model_service.py",
    "backend/api/routes.py",
    "backend/app/middleware.py",
    "backend/tests/test_routes.py",
    "ai/training/configs/baseline_frozen.yaml",
    "ai/training/configs/finetune_full.yaml",
    "ai/dataset/processed/train_manifest.csv",
    "ai/dataset/processed/val_manifest.csv",
    "ai/dataset/processed/test_manifest.csv",
)

MANIFEST_COLUMNS = ("file_path", "label", "label_id", "duration_sec", "sample_rate")
MANIFEST_PATHS = (
    "ai/dataset/processed/train_manifest.csv",
    "ai/dataset/processed/val_manifest.csv",
    "ai/dataset/processed/test_manifest.csv",
)


@dataclass
class StepResult:
    name: str
    status: str
    duration_sec: float
    details: list[str] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_command(
    name: str,
    cmd: list[str],
    *,
    extra_env: dict[str, str] | None = None,
) -> StepResult:
    start = time.perf_counter()
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    elapsed = round(time.perf_counter() - start, 3)
    stdout_tail = proc.stdout.strip().splitlines()[-8:]
    stderr_tail = proc.stderr.strip().splitlines()[-8:]
    details = [f"command: {' '.join(cmd)}", f"exit_code: {proc.returncode}"]
    if stdout_tail:
        details.append("stdout_tail:")
        details.extend(stdout_tail)
    if stderr_tail:
        details.append("stderr_tail:")
        details.extend(stderr_tail)
    return StepResult(
        name=name,
        status="pass" if proc.returncode == 0 else "fail",
        duration_sec=elapsed,
        details=details,
    )


def _check_required_files() -> StepResult:
    start = time.perf_counter()
    missing = []
    for rel in REQUIRED_FILES:
        if not (REPO_ROOT / rel).exists():
            missing.append(rel)
    elapsed = round(time.perf_counter() - start, 3)
    if missing:
        return StepResult(
            name="required_files",
            status="fail",
            duration_sec=elapsed,
            details=[f"missing: {x}" for x in missing],
        )
    return StepResult(
        name="required_files",
        status="pass",
        duration_sec=elapsed,
        details=[f"found {len(REQUIRED_FILES)} required files"],
    )


def _check_label_taxonomy() -> StepResult:
    start = time.perf_counter()
    details: list[str] = []
    status = "pass"

    if tuple(CLASS_LABELS) != EXPECTED_LABELS:
        status = "fail"
        details.append("CLASS_LABELS does not match expected 7-class taxonomy")
        details.append(f"expected: {EXPECTED_LABELS}")
        details.append(f"actual:   {tuple(CLASS_LABELS)}")
    else:
        details.append("CLASS_LABELS matches expected taxonomy")

    if NUM_CLASSES != 7:
        status = "fail"
        details.append(f"NUM_CLASSES expected 7 got {NUM_CLASSES}")

    for idx, label in enumerate(EXPECTED_LABELS):
        if LABEL2ID.get(label) != idx:
            status = "fail"
            details.append(
                f"LABEL2ID mismatch for {label}: expected {idx}, got {LABEL2ID.get(label)}"
            )
    elapsed = round(time.perf_counter() - start, 3)
    return StepResult(
        name="label_taxonomy",
        status=status,
        duration_sec=elapsed,
        details=details,
    )


def _check_training_configs() -> StepResult:
    start = time.perf_counter()
    details: list[str] = []
    status = "pass"

    for rel in ("ai/training/configs/baseline_frozen.yaml", "ai/training/configs/finetune_full.yaml"):
        path = REPO_ROOT / rel
        try:
            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:  # noqa: BLE001
            status = "fail"
            details.append(f"failed to parse {rel}: {exc}")
            continue

        for section in ("model", "training", "data", "output"):
            if section not in cfg:
                status = "fail"
                details.append(f"{rel}: missing section '{section}'")

        num_classes = cfg.get("model", {}).get("num_classes")
        if int(num_classes or -1) != 7:
            status = "fail"
            details.append(f"{rel}: model.num_classes must be 7, got {num_classes}")

        for key in ("train_manifest", "val_manifest", "test_manifest"):
            val = cfg.get("data", {}).get(key)
            if not val:
                status = "fail"
                details.append(f"{rel}: data.{key} is missing")
            else:
                details.append(f"{rel}: data.{key} -> {val}")

    elapsed = round(time.perf_counter() - start, 3)
    return StepResult(
        name="training_config_schema",
        status=status,
        duration_sec=elapsed,
        details=details,
    )


def _check_manifests() -> StepResult:
    start = time.perf_counter()
    details: list[str] = []
    status = "pass"

    for rel in MANIFEST_PATHS:
        path = REPO_ROOT / rel
        if not path.exists():
            status = "fail"
            details.append(f"{rel}: missing file")
            continue

        row_count = 0
        bad_labels = 0
        bad_label_ids = 0
        missing_local_files = 0

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                status = "fail"
                details.append(f"{rel}: empty or missing header")
                continue
            missing_cols = [c for c in MANIFEST_COLUMNS if c not in reader.fieldnames]
            if missing_cols:
                status = "fail"
                details.append(f"{rel}: missing columns {missing_cols}")
                continue

            for row in reader:
                row_count += 1
                label = str(row.get("label", "")).strip()
                label_id_raw = str(row.get("label_id", "")).strip()
                file_path = str(row.get("file_path", "")).strip()

                if label not in LABEL2ID:
                    bad_labels += 1
                    continue
                try:
                    label_id = int(label_id_raw)
                except Exception:  # noqa: BLE001
                    bad_label_ids += 1
                    continue
                if LABEL2ID[label] != label_id:
                    bad_label_ids += 1
                if file_path and not Path(file_path).exists():
                    missing_local_files += 1

        if row_count == 0:
            status = "fail"
            details.append(f"{rel}: no rows found")
        if bad_labels > 0:
            status = "fail"
            details.append(f"{rel}: invalid label values={bad_labels}")
        if bad_label_ids > 0:
            status = "fail"
            details.append(f"{rel}: label_id mismatches={bad_label_ids}")
        details.append(
            f"{rel}: rows={row_count}, missing_local_files={missing_local_files}"
        )

    elapsed = round(time.perf_counter() - start, 3)
    return StepResult(
        name="manifest_schema_and_labels",
        status=status,
        duration_sec=elapsed,
        details=details,
    )


def _overall_status(results: list[StepResult]) -> str:
    return "fail" if any(r.status == "fail" for r in results) else "pass"


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-data readiness checks.")
    parser.add_argument(
        "--report-path",
        default="ai/training/logs/pre_data_readiness_report.json",
        help="Output JSON report path (relative to repo root by default).",
    )
    parser.add_argument("--skip-lint", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    report_path = Path(args.report_path)
    if not report_path.is_absolute():
        report_path = (REPO_ROOT / report_path).resolve()

    results: list[StepResult] = []
    results.append(_check_required_files())
    results.append(_check_label_taxonomy())
    results.append(_check_training_configs())
    results.append(_check_manifests())

    if args.skip_lint:
        results.append(
            StepResult(
                name="lint",
                status="skip",
                duration_sec=0.0,
                details=["skipped by --skip-lint"],
            )
        )
    else:
        results.append(
            _run_command(
                "lint",
                [sys.executable, "-m", "flake8", "ai", "backend", "--config=.flake8"],
            )
        )

    if args.skip_tests:
        results.append(
            StepResult(
                name="tests_unit",
                status="skip",
                duration_sec=0.0,
                details=["skipped by --skip-tests"],
            )
        )
        results.append(
            StepResult(
                name="tests_integration",
                status="skip",
                duration_sec=0.0,
                details=["skipped by --skip-tests"],
            )
        )
    else:
        pytest_env = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
        results.append(
            _run_command(
                "tests_unit",
                [sys.executable, "-m", "pytest", "tests", "-q"],
                extra_env=pytest_env,
            )
        )
        results.append(
            _run_command(
                "tests_integration",
                [sys.executable, "-m", "pytest", "backend/tests", "-q"],
                extra_env=pytest_env,
            )
        )

    if args.skip_smoke:
        results.append(
            StepResult(
                name="model_smoke",
                status="skip",
                duration_sec=0.0,
                details=["skipped by --skip-smoke"],
            )
        )
    else:
        results.append(
            _run_command(
                "model_smoke",
                [sys.executable, "scripts/test_classifier_smoke.py"],
            )
        )

    overall = _overall_status(results)
    payload = {
        "generated_at_utc": _now_iso(),
        "repo_root": str(REPO_ROOT),
        "overall_status": overall,
        "steps": [asdict(x) for x in results],
    }
    _write_report(report_path, payload)

    print(f"Pre-data readiness overall status: {overall.upper()}")
    for step in results:
        print(f"- {step.name}: {step.status} ({step.duration_sec:.3f}s)")
    print(f"Report: {report_path}")

    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

