"""
Dataset Upload Script — stuttering-ai-system
=============================================
Uploads local raw .wav files to S3 under a versioned, canonical-label prefix:

    s3://<bucket>/raw/<version>/<label>/<filename>.wav

Re-runs are safe: files already in S3 with a matching ETag are skipped.

Usage:
    python scripts/upload_dataset.py --version v1.0 [--dry-run]
    python scripts/upload_dataset.py --version v1.0 [--dataset-root PATH]
    python scripts/upload_dataset.py --version v1.0 [--bucket NAME]

AWS credentials must be present in the environment or ~/.aws/config profile.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from shared.labels import CLASS_LABELS
except ModuleNotFoundError:
    # Allow running as a standalone script from outside repo root.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from shared.labels import CLASS_LABELS

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

BUCKET_DEFAULT = "stuttering-ai-data"
DATASET_ROOT_DEFAULT = "ai/dataset/raw"

# Canonical class label -> accepted local source directories.
# Supports both canonical folders and legacy naming from early dataset drops.
LABEL_SOURCE_DIRS: dict[str, tuple[str, ...]] = {
    "fluent": ("fluent", "Fluent"),
    "blocks": ("blocks", "Blocks"),
    "interjections": ("interjections", "Interjections"),
    "prolongations": ("prolongations", "Prolongations"),
    "part_word_repetition": (
        "part_word_repetition",
        "repetitions/part_word",
        "Repetitions/Part-word repetition",
    ),
    "phrase_repetition": (
        "phrase_repetition",
        "repetitions/phrase",
        "Repetitions/Phrase repetition",
    ),
    "word_repetition": (
        "word_repetition",
        "repetitions/word",
        "Repetitions/Word repetition",
    ),
}

# ── Helpers ──────────────────────────────────────────────────────────────────


def build_s3_key(version: str, label: str, filename: str) -> str:
    return f"raw/{version}/{label}/{filename}"


def local_md5(path: Path) -> str:
    """Compute MD5 in 1 MiB chunks to avoid loading the whole file at once."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def already_uploaded(
    s3_client, bucket: str, key: str, local_path: Path
) -> bool:
    """True if S3 ETag matches local MD5.

    ETag == MD5 only for single-part uploads, which is all we do here.
    Multi-part ETags have a dash suffix like '...abc-3' and won't match,
    so those will be re-uploaded — acceptable for this script's use case.
    """
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        remote_etag = head["ETag"].strip('"')
        return remote_etag == local_md5(local_path)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def collect_local_wavs(dataset_root: Path, label: str) -> list[Path]:
    """Collect .wav files for one label from canonical and legacy paths."""
    wavs: list[Path] = []
    seen: set[str] = set()

    for rel_dir in LABEL_SOURCE_DIRS[label]:
        src_dir = dataset_root / Path(rel_dir)
        if not src_dir.is_dir():
            continue

        for wav in sorted(src_dir.glob("*.wav")):
            key = str(wav.resolve())
            if key in seen:
                continue
            seen.add(key)
            wavs.append(wav)

    return wavs


def upload_class(
    s3_client,
    bucket: str,
    version: str,
    label: str,
    wavs: list[Path],
    dry_run: bool,
) -> tuple[int, int]:
    if not wavs:
        return 0, 0

    uploaded = skipped = 0
    for wav in tqdm(wavs, desc=f"  {label}", unit="file", leave=False):
        key = build_s3_key(version, label, wav.name)
        if dry_run:
            print(f"    [dry-run] s3://{bucket}/{key}")
            uploaded += 1
            continue
        if already_uploaded(s3_client, bucket, key, wav):
            skipped += 1
            continue
        s3_client.upload_file(str(wav), bucket, key)
        uploaded += 1

    return uploaded, skipped


def verify_upload(s3_client, bucket: str, version: str) -> dict[str, int]:
    paginator = s3_client.get_paginator("list_objects_v2")
    counts: dict[str, int] = {}
    for label in CLASS_LABELS:
        prefix = f"raw/{version}/{label}/"
        total = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith("/"):
                    total += 1
        counts[label] = total
    return counts


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Upload raw labeled .wav dataset to S3 under a versioned prefix."
        )
    )
    p.add_argument("--version", required=True,
                   help="Dataset version tag, e.g. v1.0")
    p.add_argument("--dataset-root", default=DATASET_ROOT_DEFAULT,
                   help=(
                       "Local dataset root directory "
                       f"(default: {DATASET_ROOT_DEFAULT})"
                   ))
    p.add_argument("--bucket", default=BUCKET_DEFAULT,
                   help=f"S3 bucket name (default: {BUCKET_DEFAULT})")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be uploaded without writing to S3")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = (repo_root / args.dataset_root).resolve()

    if not dataset_root.is_dir():
        logging.error("dataset root not found: %s", dataset_root)
        sys.exit(1)

    s3 = None if args.dry_run else boto3.client("s3")

    if args.dry_run:
        print(
            f"[dry-run] would upload to "
            f"s3://{args.bucket}/raw/{args.version}/\n"
        )
    else:
        print(f"uploading to s3://{args.bucket}/raw/{args.version}/\n")

    total_uploaded = total_skipped = 0
    failed: list[str] = []

    for label in CLASS_LABELS:
        wavs = collect_local_wavs(dataset_root, label)
        if not wavs:
            logging.warning("no local .wav files found for label=%s", label)
            continue
        try:
            up, sk = upload_class(
                s3, args.bucket, args.version, label, wavs, args.dry_run
            )
            total_uploaded += up
            total_skipped += sk
            if not args.dry_run:
                print(
                    f"  {label:<28}  uploaded={up:>4}  skipped={sk:>4}"
                )
        except (BotoCoreError, ClientError) as exc:
            logging.error("upload failed for %s: %s", label, exc)
            failed.append(label)

    if failed:
        logging.error("classes that failed: %s", ", ".join(failed))
        sys.exit(1)

    if args.dry_run:
        print(f"\n[dry-run] {total_uploaded} files would be uploaded")
        return

    print(
        f"\n  total  uploaded={total_uploaded}  skipped={total_skipped}"
    )

    print("\nverifying S3 object counts …")
    counts = verify_upload(s3, args.bucket, args.version)
    for label, n in counts.items():
        print(f"  {label:<28}  {n:>5} objects")
    grand_total = sum(counts.values())
    print(
        f"\n  grand total: {grand_total} objects at "
        f"s3://{args.bucket}/raw/{args.version}/"
    )


if __name__ == "__main__":
    main()
