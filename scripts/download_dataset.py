"""
Dataset Download Script — stuttering-ai-system
===============================================
Syncs a versioned S3 dataset prefix to a local directory:

    s3://<bucket>/raw/<version>/<label>/ -> <target>/<label>/

Existing files are skipped by default (size check). Pass --verify to do
a full MD5 comparison instead.

Usage:
    python scripts/download_dataset.py --version v1.0
    python scripts/download_dataset.py --version v1.0 [--target PATH]
    python scripts/download_dataset.py --version v1.0 [--bucket NAME]

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

# repo root must be on sys.path so `shared` is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.labels import CLASS_LABELS  

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

BUCKET_DEFAULT = "stuttering-ai-data"
TARGET_DEFAULT = "ai/dataset/raw"

# ── Helpers ──────────────────────────────────────────────────────────────────


def resolve_target(target_arg: str) -> Path:
    p = Path(target_arg)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parents[1] / p).resolve()


def local_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_class(
    s3_client,
    bucket: str,
    version: str,
    label: str,
    target_dir: Path,
    verify: bool,
) -> tuple[int, int]:
    target_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"raw/{version}/{label}/"
    paginator = s3_client.get_paginator("list_objects_v2")
    objects = [
        obj
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
        # S3 sometimes emits a zero-byte "folder" key ending in / — skip it
        if not obj["Key"].endswith("/")
    ]

    if not objects:
        logging.warning("no objects found at s3://%s/%s", bucket, prefix)
        return 0, 0

    downloaded = skipped = 0
    for obj in tqdm(objects, desc=f"  {label}", unit="file", leave=False):
        filename = Path(obj["Key"]).name
        local_path = target_dir / filename

        if local_path.exists():
            if verify:
                # slower but guarantees integrity after a partial download
                if local_md5(local_path) == obj["ETag"].strip('"'):
                    skipped += 1
                    continue
            elif local_path.stat().st_size == obj["Size"]:
                skipped += 1
                continue

        s3_client.download_file(bucket, obj["Key"], str(local_path))
        downloaded += 1

    return downloaded, skipped


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a versioned raw dataset from S3 to local disk."
    )
    p.add_argument("--version", required=True,
                   help="Dataset version tag, e.g. v1.0")
    p.add_argument("--target", default=TARGET_DEFAULT,
                   help=(
                       "Local destination directory "
                       f"(default: {TARGET_DEFAULT})"
                   ))
    p.add_argument("--bucket", default=BUCKET_DEFAULT,
                   help=f"S3 bucket name (default: {BUCKET_DEFAULT})")
    p.add_argument("--verify", action="store_true",
                   help="Check existing files via MD5 rather than file size")
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

    target_root = resolve_target(args.target)
    print(
        f"syncing s3://{args.bucket}/raw/{args.version}/ -> {target_root}\n"
    )

    s3 = boto3.client("s3")  # creds from env vars / ~/.aws/config

    total_downloaded = total_skipped = 0
    failed: list[str] = []

    for label in CLASS_LABELS:
        label_dir = target_root / label
        try:
            dl, sk = download_class(
                s3, args.bucket, args.version, label, label_dir, args.verify
            )
            total_downloaded += dl
            total_skipped += sk
            print(
                f"  {label:<28}  downloaded={dl:>4}  skipped={sk:>4}"
            )
        except (BotoCoreError, ClientError) as exc:
            logging.error("download failed for %s: %s", label, exc)
            failed.append(label)

    if failed:
        logging.error("classes that failed: %s", ", ".join(failed))
        sys.exit(1)

    print(
        f"\n  total  downloaded={total_downloaded}"
        f"  already up-to-date={total_skipped}"
    )


if __name__ == "__main__":
    main()
