"""
Download a versioned model artifact from S3 and verify its integrity.

Usage:
    python scripts/download_model.py \
        --version v1.0 \
        --output_dir /tmp/model_artifacts

The script compares each file's MD5 against the S3 ETag.  For objects
uploaded in a single part (< 5 GB, which covers all normal model files)
the ETag IS the MD5 hex digest — so the comparison is exact.

Credentials come from environment variables only:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""

import argparse
import hashlib
import sys
from pathlib import Path
from dotenv import load_dotenv

import boto3
from botocore.exceptions import BotoCoreError, ClientError

load_dotenv()

BUCKET = "stuttering-ai-models"
ARTIFACTS = ["model_inference.pt", "config.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a versioned model artifact from S3."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Version to download, e.g. v1.0.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Local directory where artifacts will be written.",
    )
    return parser.parse_args()


def md5_of_file(path: Path) -> str:
    """Return the MD5 hex digest of a local file (matches single-part S3 ETags)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_etag_quotes(etag: str) -> str:
    """S3 returns ETags wrapped in double-quotes — strip them for comparison."""
    return etag.strip('"')


def download_and_verify(s3_client, version: str, output_dir: Path) -> None:
    """
    Download every artifact for the given version and verify each one.
    Exits with a non-zero code if any verification fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for filename in ARTIFACTS:
        s3_key = f"{version}/{filename}"
        local_path = output_dir / filename

        # Grab the ETag before downloading so we have something to compare against
        try:
            head = s3_client.head_object(Bucket=BUCKET, Key=s3_key)
        except ClientError as exc:
            print(
                f"ERROR: Could not stat s3://{BUCKET}/{s3_key}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

        expected_md5 = strip_etag_quotes(head["ETag"])

        print(f"  Downloading s3://{BUCKET}/{s3_key} → {local_path}")
        try:
            s3_client.download_file(Bucket=BUCKET, Key=s3_key, Filename=str(local_path))
        except ClientError as exc:
            print(f"ERROR: Download failed for {filename}: {exc}", file=sys.stderr)
            sys.exit(1)

        actual_md5 = md5_of_file(local_path)

        if actual_md5 != expected_md5:
            # Don't silently leave a corrupt file on disk
            local_path.unlink(missing_ok=True)
            failures.append(
                f"{filename}: expected MD5={expected_md5}, got {actual_md5}"
            )
            print(f"  ✗ {filename}  HASH MISMATCH — file removed")
        else:
            print(f"  ✓ {filename}  MD5={actual_md5[:12]}… verified")

    if failures:
        print("\nERROR: The following files failed hash verification:", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    try:
        s3 = boto3.client("s3")
    except BotoCoreError as exc:
        print(f"ERROR: Could not initialise S3 client: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDownloading version '{args.version}' from s3://{BUCKET}/")
    download_and_verify(s3, args.version, output_dir)
    print(f"\nAll artifacts verified and saved to {output_dir}")


if __name__ == "__main__":
    main()