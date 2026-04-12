"""
Lightweight hyperparameter search: short training runs, CSV log, best YAML export.

Usage:
  python ai/training/hparam_search.py --search_space ai/training/configs/search_space.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai.training.train import _load_config, run_training  # noqa: E402

_GRID_ORDER = ("learning_rate", "batch_size", "dropout_rate", "freeze_encoder")


def _resolve_path(cwd: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (cwd / path).resolve()


def _load_search_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"Search space must be a mapping: {path}")
    return spec


def _build_combinations(
    spec: dict[str, Any],
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    grid = spec["grid"]
    limited = bool(spec.get("limited_compute", False))
    sampling = spec.get("sampling") or {}
    method = str(sampling.get("method", "grid")).lower()
    seed = int(sampling.get("random_seed", 42))

    if limited:
        combos: list[tuple[Any, Any, Any, Any]] = []
        for lr in grid["learning_rate"]:
            for fe in grid["freeze_encoder"]:
                combos.append(
                    (
                        float(lr),
                        int(base["training"]["batch_size"]),
                        float(base["model"]["dropout_rate"]),
                        bool(fe),
                    )
                )
    else:
        lists = [grid[k] for k in _GRID_ORDER]
        combos = list(itertools.product(*lists))
        combos = [
            (
                float(t[0]),
                int(t[1]),
                float(t[2]),
                bool(t[3]),
            )
            for t in combos
        ]

    keyed = [dict(zip(_GRID_ORDER, c)) for c in combos]

    if method == "random":
        n_trials = sampling.get("n_trials")
        if n_trials is None:
            raise ValueError("sampling.n_trials is required when sampling.method is 'random'")
        n = int(n_trials)
        rng = random.Random(seed)
        rng.shuffle(keyed)
        keyed = keyed[: min(n, len(keyed))]
    elif method != "grid":
        raise ValueError(f"Unknown sampling.method {method!r}; use 'grid' or 'random'")

    return keyed


def _apply_combo(raw: dict[str, Any], combo: dict[str, Any], *, budget_epochs: int) -> None:
    raw["training"]["learning_rate"] = float(combo["learning_rate"])
    raw["training"]["batch_size"] = int(combo["batch_size"])
    raw["model"]["dropout_rate"] = float(combo["dropout_rate"])
    raw["model"]["freeze_encoder"] = bool(combo["freeze_encoder"])
    raw["training"]["num_epochs"] = int(budget_epochs)


def _hyperparams_json(combo: dict[str, Any]) -> str:
    payload = {k: combo[k] for k in _GRID_ORDER}
    return json.dumps(payload, sort_keys=True)


def _print_ranked(rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        rows,
        key=lambda r: float(r["val_macro_f1"]),
        reverse=True,
    )
    print()
    print("=== Hyperparameter search (ranked by val_macro_f1) ===")
    hdr = f"{'rank':>4}  {'experiment_name':<22}  {'val_macro_f1':>14}  {'val_loss':>10}  hyperparams"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(ranked, start=1):
        print(
            f"{i:>4}  {r['experiment_name']:<22}  "
            f"{float(r['val_macro_f1']):>14.6f}  {float(r['val_loss']):>10.6f}  "
            f"{r['hyperparams']}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Short-budget hyperparameter search for training.")
    parser.add_argument(
        "--search_space",
        type=str,
        required=True,
        help="YAML file defining base config, grid, sampling, and output paths.",
    )
    args = parser.parse_args()

    cwd = Path.cwd().resolve()
    spec_path = Path(args.search_space).resolve()
    spec = _load_search_spec(spec_path)

    base_path = _resolve_path(cwd, spec["base_config"])
    base_pristine = _load_config(base_path)

    budget_epochs = int(spec.get("budget_epochs", 4))
    if budget_epochs < 3 or budget_epochs > 5:
        # Allow override but warn if outside suggested band
        print(
            f"Note: budget_epochs={budget_epochs} (typical search uses 3–5).",
            flush=True,
        )

    full_epochs = spec.get("full_training_epochs")
    if full_epochs is None:
        full_epochs = int(base_pristine["training"]["num_epochs"])
    else:
        full_epochs = int(full_epochs)

    out_cfg = spec.get("output") or {}
    results_csv = _resolve_path(cwd, out_cfg.get("results_csv", "ai/training/hparam_results.csv"))
    best_config_path = _resolve_path(cwd, out_cfg.get("best_config", "ai/training/configs/best_config.yaml"))
    best_experiment_name = str(out_cfg.get("best_experiment_name", "hparam_best"))

    combinations = _build_combinations(spec, base_pristine)
    if not combinations:
        raise RuntimeError("No hyperparameter combinations to run (check search_space).")

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("experiment_name", "hyperparams", "val_macro_f1", "val_loss")
    rows_out: list[dict[str, Any]] = []

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial_idx, combo in enumerate(combinations):
            trial_cfg = copy.deepcopy(base_pristine)
            experiment_name = f"hparam_{trial_idx:04d}"
            trial_cfg["output"]["experiment_name"] = experiment_name
            _apply_combo(trial_cfg, combo, budget_epochs=budget_epochs)

            hp_json = _hyperparams_json(combo)
            print(
                f"\n--- Trial {trial_idx + 1}/{len(combinations)}: {experiment_name} ---\n"
                f"hyperparams={hp_json}\n",
                flush=True,
            )

            metrics = run_training(
                trial_cfg,
                cwd=cwd,
                config_path=None,
                verbose=True,
            )
            val_macro_f1 = metrics["best_val_macro_f1"]
            val_loss = metrics["val_loss_at_best_f1"]

            row = {
                "experiment_name": experiment_name,
                "hyperparams": hp_json,
                "val_macro_f1": f"{val_macro_f1:.8f}",
                "val_loss": f"{val_loss:.8f}",
            }
            writer.writerow(row)
            f.flush()
            rows_out.append(
                {
                    "experiment_name": experiment_name,
                    "hyperparams": hp_json,
                    "val_macro_f1": val_macro_f1,
                    "val_loss": val_loss,
                    "combo": combo,
                }
            )

    best_row = max(rows_out, key=lambda r: float(r["val_macro_f1"]))
    winner_cfg = copy.deepcopy(base_pristine)
    winner_cfg["output"]["experiment_name"] = best_experiment_name
    _apply_combo(winner_cfg, best_row["combo"], budget_epochs=full_epochs)

    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    with best_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            winner_cfg,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    _print_ranked(rows_out)
    print(f"Wrote results: {results_csv}")
    print(
        f"Best trial by val_macro_f1: {best_row['experiment_name']} "
        f"(val_macro_f1={best_row['val_macro_f1']:.6f}, val_loss={best_row['val_loss']:.6f})",
        flush=True,
    )
    print(f"Wrote best config for full training ({full_epochs} epochs): {best_config_path}")


if __name__ == "__main__":
    main()
