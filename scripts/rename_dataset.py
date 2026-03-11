"""
Dataset Rename & Clean Script — stuttering-ai-system
=====================================================
Reads dataset_inventory.csv produced by audit_dataset.py and:
  1. Renames nonstandard files (strips "Copy of " prefix, replaces spaces with _)
  2. Archives cross-class duplicate files to {dataset_root}/_duplicates/

Safe by default: runs in DRY-RUN mode unless --execute is passed.

Usage:
    python scripts/rename_dataset.py --dry-run          # preview only (default)
    python scripts/rename_dataset.py --execute           # apply changes
    python scripts/rename_dataset.py --execute --log-level DEBUG
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

SCRIPT_VERSION = "1.0.0"
_COPY_OF_PREFIX = re.compile(r"^Copy of\s+", re.IGNORECASE)

# ── Shared utilities (mirrors audit_dataset.py pattern) ───────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _to_abs_path(raw: str, *, must_exist: bool = False) -> Path:
    p = (Path(raw) if Path(raw).is_absolute() else _project_root() / raw).resolve()
    if must_exist and not p.exists():
        sys.exit(f"[ERROR] Path not found: {p}")
    return p

def setup_logging(level: str, log_file: Path) -> logging.Logger:
    log = logging.getLogger("rename_dataset")
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log

# ── Name cleaning (SRP) ────────────────────────────────────────────────────────

def _clean_name(filename: str) -> str:
    """Strip 'Copy of ' prefix then replace spaces with underscores."""
    name = _COPY_OF_PREFIX.sub("", filename).strip()
    return name.replace(" ", "_")

def _unique_target(folder: Path, name: str) -> Path:
    """Return a non-colliding target path; appends _1, _2 … if needed."""
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate = folder / name
    counter = 1
    while candidate.exists():
        candidate = folder / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate

# ── Target loaders (SRP — read CSV, build action lists) ───────────────────────

def _load_rename_targets(csv_path: Path) -> list[dict]:
    """Return list of {src, dst} for all nonstandard-filename files."""
    df = pd.read_csv(csv_path)
    targets = []
    for _, row in df[df["has_nonstandard_filename"] == True].iterrows():
        src = Path(row["absolute_path"])
        if not src.exists():
            continue
        clean = _clean_name(src.name)
        dst = _unique_target(src.parent, clean)
        targets.append({"src": src, "dst": dst})
    return targets

def _load_duplicate_targets(csv_path: Path) -> list[dict]:
    """
    Return list of {src, archive_dst} for files to archive.

    For each cross-class duplicate group: keep the copy in the smallest class
    (fewest files), archive the rest. Same-class duplicates are skipped
    (user should resolve manually after rename).
    """
    df = pd.read_csv(csv_path)

    # Count files per class for tie-breaking
    class_sizes = df.groupby("class_label").size().to_dict()

    groups: dict[str, list] = defaultdict(list)
    for _, row in df[df["is_duplicate"] == True].iterrows():
        groups[row["md5_hash"]].append(row)

    to_archive = []
    for md5, rows in groups.items():
        labels = {r["class_label"] for r in rows}
        if len(labels) == 1:
            # Same-class duplicate — skip; audit will still flag but not our job here
            continue
        # Cross-class: keep in smallest class, archive from largest
        sorted_rows = sorted(rows, key=lambda r: class_sizes.get(r["class_label"], 0))
        for row in sorted_rows[1:]:  # everything except the smallest-class copy
            src = Path(row["absolute_path"])
            if src.exists():
                to_archive.append({"src": src, "class_label": row["class_label"]})

    return to_archive

# ── Executors (SRP — perform or simulate the file operations) ─────────────────

def rename_files(targets: list[dict], dry_run: bool) -> int:
    """Rename src → dst for each target. Returns count of files renamed."""
    log = logging.getLogger("rename_dataset")
    count = 0
    for t in targets:
        src, dst = t["src"], t["dst"]
        log.info("%s  →  %s", src.name, dst.name)
        if not dry_run:
            src.rename(dst)
        count += 1
    return count

def archive_duplicates(targets: list[dict], archive_dir: Path, dry_run: bool) -> int:
    """Move duplicate files to archive_dir. Returns count archived."""
    log = logging.getLogger("rename_dataset")
    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for t in targets:
        src = t["src"]
        dst = _unique_target(archive_dir, src.name)
        log.info("ARCHIVE  %s (%s)  →  _duplicates/%s", src.name, t["class_label"], dst.name)
        if not dry_run:
            src.rename(dst)
        count += 1
    return count

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root",   default="ai/dataset/raw",                       help="Dataset root (default: ai/dataset/raw)")
    p.add_argument("--inventory-csv",  default="ai/dataset/metadata/dataset_inventory.csv", help="CSV from audit_dataset.py")
    p.add_argument("--log-level",      default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run",  dest="dry_run", action="store_true",  default=True,  help="Preview only — make no changes (default)")
    mode.add_argument("--execute",  dest="dry_run", action="store_false",                help="Apply renames and archiving")
    return p.parse_args()

# ── Orchestration ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    root    = _to_abs_path(args.dataset_root, must_exist=True)
    csv     = _to_abs_path(args.inventory_csv, must_exist=True)
    log_dir = csv.parent
    log     = setup_logging(args.log_level, log_dir / "rename.log")

    mode_label = "DRY RUN — no files will be changed" if args.dry_run else "EXECUTE — files will be modified"
    log.info("rename_dataset v%s | %s | root=%s", SCRIPT_VERSION, mode_label, root)

    renames    = _load_rename_targets(csv)
    duplicates = _load_duplicate_targets(csv)

    if not renames and not duplicates:
        log.info("Nothing to do — dataset is already clean.")
        return

    log.info("Renames planned : %d", len(renames))
    log.info("Archives planned: %d", len(duplicates))

    if args.dry_run:
        print(f"\n{'─'*60}")
        print(f"DRY RUN — pass --execute to apply changes")
        print(f"{'─'*60}\n")

    renamed  = rename_files(renames, dry_run=args.dry_run)
    archived = archive_duplicates(duplicates, archive_dir=root / "_duplicates", dry_run=args.dry_run)

    log.info("Done: %d renamed, %d archived.", renamed, archived)
    if not args.dry_run:
        log.info("Next: run  python3 scripts/audit_dataset.py  to regenerate CSV and report.")

if __name__ == "__main__":
    main()
