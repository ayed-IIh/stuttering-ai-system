# S3 Architecture Documentation — Stuttering AI Project

**Project:** Stuttering AI

**Document:** S3 Bucket Architecture

**Version:** 1.0

**Last Updated:** 2026-03-25

**Reference:** WAE-05

---

## Table of Contents

1. [Naming Conventions](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#naming-conventions)
2. [Bucket Overview](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#bucket-overview)
3. [Bucket 1: stuttering-ai-data](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#bucket-1-stuttering-ai-data)
4. [Bucket 2: stuttering-ai-models](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#bucket-2-stuttering-ai-models)
5. [Bucket 3: stuttering-ai-logs](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#bucket-3-stuttering-ai-logs)
6. [How to Apply Bucket Policies (AWS Console)](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#how-to-apply-bucket-policies-aws-console)

---

## Naming Conventions

All buckets follow the pattern:

```
stuttering-ai-<purpose>
```

| Segment             | Rule                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Prefix              | `stuttering-ai-`— identifies the project                                                                               |
| Purpose             | Lowercase, hyphen-separated descriptor:`data`,`models`,`logs`                                                       |
| Global uniqueness   | S3 bucket names are globally unique; no account ID suffix is required as the names above were reserved at bucket creation |
| No uppercase        | S3 DNS-compliant names are always lowercase                                                                               |
| No trailing hyphens | Prohibited by AWS naming rules                                                                                            |

---

## Bucket Overview

| Bucket Name              | Purpose                  | Versioning | Encryption       | Public Access | Lifecycle                  |
| ------------------------ | ------------------------ | ---------- | ---------------- | ------------- | -------------------------- |
| `stuttering-ai-data`   | Raw & processed datasets | Enabled    | SSE-S3 (AES-256) | Fully blocked | None                       |
| `stuttering-ai-models` | Trained model artifacts  | Enabled    | SSE-S3 (AES-256) | Fully blocked | None                       |
| `stuttering-ai-logs`   | Application log archival | Disabled   | SSE-S3 (AES-256) | Fully blocked | Glacier @90d, Expire @365d |

---

## Bucket 1: `stuttering-ai-data`

### Purpose

Stores all raw audio/text datasets ingested by the data pipeline and processed/cleaned versions produced by preprocessing scripts. Acts as the single source of truth for training data.

### Folder Structure

```
stuttering-ai-data/
├── raw/          ← ingest landing zone (pipeline role has read/write here)
└── processed/    ← cleaned outputs (future use; no IAM grant yet)
```

### Settings

| Setting                 | Value                                                           |
| ----------------------- | --------------------------------------------------------------- |
| Access                  | Private                                                         |
| Block all public access | ✅ Enabled (all four sub-settings ON)                           |
| Versioning              | ✅ Enabled                                                      |
| Default encryption      | SSE-S3 (`AES256`)                                             |
| Bucket key              | Enabled (reduces SSE-KMS request costs; compatible with SSE-S3) |
| Object ownership        | Bucket owner enforced (ACLs disabled)                           |

### Bucket Policy

**File:** `docs/aws/bucket_policies/stuttering-ai-data-bucket-policy.json`

The policy grants access exclusively to `stuttering-ai-data-pipeline-role` (WAE-06 / Ali). No wildcard principals or actions are used. Access is scoped to:

* `s3:PutObject` → `arn:aws:s3:::stuttering-ai-data/raw/*`
* `s3:GetObject` → `arn:aws:s3:::stuttering-ai-data/raw/*`
* `s3:ListBucket` → `arn:aws:s3:::stuttering-ai-data` with `s3:prefix` condition `raw/*`

A `Deny` statement blocks all other principals from any `s3:*` action on the bucket and its contents.

### Encryption

Server-Side Encryption with Amazon S3-managed keys (SSE-S3). All objects are automatically encrypted at rest using AES-256. No customer-managed KMS key is required at this stage; encryption is transparent and requires no client-side configuration.

### Versioning

Enabled. All overwrite and delete operations on objects under `raw/` create new versions rather than destroying data. This protects against accidental overwrites by the data pipeline. Lifecycle rules for non-current version expiry can be added when storage costs become a concern.

### Lifecycle Rules

None defined. Revisit after 6 months of pipeline operation to add non-current version expiry if storage costs grow.

---

## Bucket 2: `stuttering-ai-models`

### Purpose

Stores all trained model artifacts produced by the training pipeline (checkpoints, final weights, tokeniser configs, evaluation metrics). Consumed read-only by the backend API/EC2 instances at inference time.

### Folder Structure

```
stuttering-ai-models/
└── models/    ← model artifacts (write-role path restriction)
```

### Settings

| Setting                 | Value                                 |
| ----------------------- | ------------------------------------- |
| Access                  | Private                               |
| Block all public access | ✅ Enabled (all four sub-settings ON) |
| Versioning              | ✅ Enabled                            |
| Default encryption      | SSE-S3 (`AES256`)                   |
| Bucket key              | Enabled                               |
| Object ownership        | Bucket owner enforced (ACLs disabled) |

### Bucket Policy

**File:** `docs/aws/bucket_policies/stuttering-ai-models-bucket-policy.json`

Two roles are granted access (WAE-06):

| Role                               | Actions                            | Resource Scope        |
| ---------------------------------- | ---------------------------------- | --------------------- |
| `stuttering-ai-model-read-role`  | `s3:GetObject`,`s3:ListBucket` | All objects; no write |
| `stuttering-ai-model-write-role` | `s3:PutObject`                   | `models/*`path only |

A `Deny` statement blocks all principals not matching either of these two role ARNs from any `s3:*` action.

### Encryption

SSE-S3 (AES-256). Identical to `stuttering-ai-data`. All model artifacts encrypted at rest automatically.

### Versioning

Enabled. Every model upload creates a new version, allowing rollback to a prior checkpoint if a regression is introduced. The `stuttering-ai-model-write-role` cannot delete objects or versions; deletions require admin-level access.

### Lifecycle Rules

None at this time. Consider adding non-current version expiry (e.g., keep last 5 versions) once the number of training runs grows.

---

## Bucket 3: `stuttering-ai-logs`

### Purpose

Archival store for application logs generated by the backend service, data pipeline, and inference API. Not intended for live log querying — use CloudWatch Logs for that. This bucket provides long-term, low-cost retention for audit and compliance purposes.

### Folder Structure

```
stuttering-ai-logs/
└── <service>/YYYY/MM/DD/    ← recommended key prefix convention for log files
```

### Settings

| Setting                 | Value                                                  |
| ----------------------- | ------------------------------------------------------ |
| Access                  | Private                                                |
| Block all public access | ✅ Enabled (all four sub-settings ON)                  |
| Versioning              | Disabled (logs are write-once; versioning unnecessary) |
| Default encryption      | SSE-S3 (`AES256`)                                    |
| Bucket key              | Enabled                                                |
| Object ownership        | Bucket owner enforced (ACLs disabled)                  |

### Bucket Policy

**File:** `docs/aws/bucket_policies/stuttering-ai-logs-bucket-policy.json`

Grants `stuttering-ai-data-pipeline-role` (the only role that currently writes logs) the following:

* `s3:PutObject` → `arn:aws:s3:::stuttering-ai-logs/*`
* `s3:ListBucket` → `arn:aws:s3:::stuttering-ai-logs`

All other principals are denied via an explicit `Deny` statement.

> **Note:** If additional services (e.g., a separate backend role) need to write logs, add their ARNs to both the `Allow` statements and the `Deny` condition's exception list before deployment.

### Encryption

SSE-S3 (AES-256). Log files encrypted at rest automatically.

### Versioning

Disabled. Log files are immutable once written and versioning would double storage costs without benefit. The lifecycle rule below ensures cost control.

### Lifecycle Rules

Two lifecycle rules are configured on this bucket:

| Rule                    | Action                                                        | Days                    |
| ----------------------- | ------------------------------------------------------------- | ----------------------- |
| `TransitionToGlacier` | Transition objects to S3 Glacier Flexible Retrieval           | After**90 days**  |
| `ExpireObjects`       | Permanently delete objects (and incomplete multipart uploads) | After**365 days** |

**How to configure in the AWS Console (S3 → Bucket → Management → Lifecycle rules):**

1. Open `stuttering-ai-logs` → **Management** tab → **Lifecycle rules** →  **Create lifecycle rule** .
2. **Rule name:** `LogsArchiveAndExpire`
3. **Scope:** Apply to all objects in the bucket.
4. **Lifecycle rule actions:** Check both:
   * ✅ Transition current versions of objects between storage classes
   * ✅ Expire current versions of objects
5. **Transition:**
   * Storage class: **S3 Glacier Flexible Retrieval**
   * Days after object creation: **90**
6. **Expiration:**
   * Days after object creation: **365**
7. Save the rule.

---

## How to Apply Bucket Policies (AWS Console)

> Repeat these steps for each of the three buckets using the corresponding JSON file.

1. Sign in to the **AWS Management Console** → navigate to  **S3** .
2. Click the bucket name (e.g., `stuttering-ai-data`).
3. Go to the **Permissions** tab.
4. Under  **Bucket policy** , click  **Edit** .
5. Replace `ACCOUNT_ID` in the JSON file with your 12-digit AWS Account ID.
6. Paste the full contents of the corresponding policy JSON file.
7. Click  **Save changes** .

> **Important:** The `Deny` statement in each policy uses `ArnNotEquals` / `ArnNotLike` conditions. After saving, verify access immediately by assuming the intended role and running a test operation. An incorrect ARN will lock out all access including your own admin session (admin access via the root account is always preserved).
