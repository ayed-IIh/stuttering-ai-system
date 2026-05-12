"""
Upload a trained model artifact (model_inference.pt + config.json) to S3.

Usage:
    python scripts/upload_model.py \
        --checkpoint_path exports/v1.0 \
        --version v1.0 \
        --experiment_name stuttering_wav2vec_run3 \
        --val_f1 0.873

Credentials come exclusively from environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
Never pass them as arguments or hardcode them here.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

import boto3
from botocore.exceptions import BotoCoreError, ClientError

load_dotenv()

BUCKET = "stuttering-ai-models"
REQUIRED_FILES = ["model_inference.pt", "config.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a model inference artifact to S3."
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Local directory containing model_inference.pt and config.json.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Semantic version string, e.g. v1.0. Used as the S3 key prefix.",
    )
    parser.add_argument(
        "--experiment_name",
        required=True,
        help="Human-readable experiment label stored as an S3 object tag.",
    )
    parser.add_argument(
        "--val_f1",
        required=True,
        type=float,
        help="Validation F1 score stored as an S3 object tag.",
    )
    return parser.parse_args()


def sha256_of_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_tags(experiment_name: str, val_f1: float) -> list[dict]:
    """Return the S3 tag set we attach to every uploaded object."""
    return [
        {"Key": "experiment_name", "Value": experiment_name},
        # Store as string — S3 tag values must be strings
        {"Key": "val_f1", "Value": f"{val_f1:.6f}"},
        {"Key": "upload_date", "Value": datetime.now(timezone.utc).isoformat()},
    ]


def upload_file(
    s3_client,
    local_path: Path,
    s3_key: str,
    tags: list[dict],
) -> None:
    """Upload a single file and attach tags. Raises on any S3 error."""
    tag_str = "&".join(f"{t['Key']}={t['Value']}" for t in tags)

    print(f"  Uploading {local_path.name} → s3://{BUCKET}/{s3_key}")
    s3_client.upload_file(
        Filename=str(local_path),
        Bucket=BUCKET,
        Key=s3_key,
        ExtraArgs={"Tagging": tag_str},
    )

    # Quick sanity check — confirm the object is actually there
    head = s3_client.head_object(Bucket=BUCKET, Key=s3_key)
    local_sha = sha256_of_file(local_path)
    print(f"  ✓ {local_path.name}  ETag={head['ETag']}  local-sha256={local_sha[:12]}…")


def validate_source_dir(source: Path) -> None:
    """Fail fast if any expected artifact is missing from the export directory."""
    missing = [f for f in REQUIRED_FILES if not (source / f).exists()]
    if missing:
        print(f"ERROR: Missing artifact(s) in {source}: {missing}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    source = Path(args.checkpoint_path)

    validate_source_dir(source)

    # boto3 picks up credentials from env vars automatically — nothing to configure here
    try:
        s3 = boto3.client("s3")
    except BotoCoreError as exc:
        print(f"ERROR: Could not initialise S3 client: {exc}", file=sys.stderr)
        sys.exit(1)

    tags = build_tags(args.experiment_name, args.val_f1)
    print(f"\nUploading artifacts for version '{args.version}' to s3://{BUCKET}/")

    try:
        for filename in REQUIRED_FILES:
            s3_key = f"{args.version}/{filename}"
            upload_file(s3, source / filename, s3_key, tags)
    except ClientError as exc:
        print(f"ERROR: S3 upload failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll artifacts uploaded successfully under s3://{BUCKET}/{args.version}/")


if __name__ == "__main__":
    main()