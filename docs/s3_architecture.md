
# S3 Architecture Documentation — Stuttering AI Project

---

## Table of Contents

1. [Overview](#overview)
2. [Naming Conventions](#naming-conventions)
3. [Buckets Summary](#buckets-summary)
4. [Bucket Details](#bucket-details)
   - [1. stuttering-ai-data](#bucket-1-stuttering-ai-data)
   - [2. stuttering-ai-models](#bucket-2-stuttering-ai-models)
   - [3. stuttering-ai-logs](#bucket-3-stuttering-ai-logs)
5. [Bucket Policies Application (AWS Console)](#bucket-policies-application-aws-console)

---

## Overview

This document describes the S3 architecture for the **Stuttering AI Project**. It includes:

- Naming conventions
- Purpose and folder structure of each bucket
- Versioning, encryption, and lifecycle configuration
- Least-privilege bucket policies for IAM roles defined in **WAE-06**
- How to apply bucket policies in the AWS Console

All buckets follow AWS best practices: private by default, encrypted at rest, and all public access blocked.

---

## Naming Conventions

All buckets follow the pattern:

stuttering-ai-`<purpose>`


| Segment           | Rule                                                                  |
| ----------------- | --------------------------------------------------------------------- |
| Prefix            | `stuttering-ai-` — identifies the project                          |
| Purpose           | Lowercase, hyphen-separated descriptor:`data`, `models`, `logs` |
| Global uniqueness | S3 bucket names are globally unique; no account ID suffix required    |
| Uppercase         | Not allowed                                                           |
| Trailing hyphens  | Prohibited by AWS rules                                               |

---

## Buckets Summary

| Bucket Name              | Purpose                  | Versioning | Encryption       | Public Access | Lifecycle                            |
| ------------------------ | ------------------------ | ---------- | ---------------- | ------------- | ------------------------------------ |
| `stuttering-ai-data`   | Raw & processed datasets | Enabled    | SSE-S3 (AES-256) | Blocked       | None                                 |
| `stuttering-ai-models` | Trained model artifacts  | Enabled    | SSE-S3 (AES-256) | Blocked       | None                                 |
| `stuttering-ai-logs`   | Application log archival | Disabled   | SSE-S3 (AES-256) | Blocked       | Transition: Glacier 90d, Expire 365d |

> ✅ All buckets are **private**, **encrypted**, and follow **least-privilege policies**.

---

## Bucket Details

### Bucket 1: `stuttering-ai-data`

**Purpose:**
Stores all raw and processed datasets ingested by the pipeline. Single source of truth for training.

**Folder Structure:**

stuttering-ai-data/

├── raw/       ← ingest landing zone (pipeline role read/write)

└── processed/ ← cleaned outputs (future use)


**Settings:**

| Setting                 | Value                                 |
| ----------------------- | ------------------------------------- |
| Access                  | Private                               |
| Block all public access | ✅ Enabled                            |
| Versioning              | ✅ Enabled                            |
| Default encryption      | SSE-S3 (AES-256)                      |
| Bucket key              | Enabled                               |
| Object ownership        | Bucket owner enforced (ACLs disabled) |

**Bucket Policy:**

- **Role:** `stuttering-ai-data-pipeline-role`
  - `s3:PutObject` → `stuttering-ai-data/raw/*`
  - `s3:GetObject` → `stuttering-ai-data/raw/*`
  - `s3:ListBucket` → `stuttering-ai-data` (condition: `prefix=raw/*`)
- All other principals are implicitly denied.

**Encryption:** SSE-S3 (AES-256)

**Versioning:** Enabled

**Lifecycle Rules:** None

---

### Bucket 2: `stuttering-ai-models`

**Purpose:**
Stores trained model artifacts (checkpoints, weights, tokeniser configs, metrics).

**Folder Structure:**

stuttering-ai-models/

└── models/ ← artifacts only

**Settings:**

| Setting                 | Value                                 |
| ----------------------- | ------------------------------------- |
| Access                  | Private                               |
| Block all public access | ✅ Enabled                            |
| Versioning              | ✅ Enabled                            |
| Default encryption      | SSE-S3 (AES-256)                      |
| Bucket key              | Enabled                               |
| Object ownership        | Bucket owner enforced (ACLs disabled) |

**Bucket Policy:**

- **Role:** `stuttering-ai-model-read-role`
  - `s3:GetObject` → `stuttering-ai-models/*`
  - `s3:ListBucket` → `stuttering-ai-models`
- **Role:** `stuttering-ai-model-write-role`
  - `s3:PutObject` → `stuttering-ai-models/models/*`
- All other principals are implicitly denied.

**Encryption:** SSE-S3 (AES-256)

**Versioning:** Enabled

**Lifecycle Rules:** None

---

### Bucket 3: `stuttering-ai-logs`

**Purpose:**
Archival store for backend, pipeline, and inference logs.

**Folder Structure:**

stuttering-ai-logs/

└── `<service>`/YYYY/MM/DD/


**Settings:**

| Setting                 | Value                                 |
| ----------------------- | ------------------------------------- |
| Access                  | Private                               |
| Block all public access | ✅ Enabled                            |
| Versioning              | Disabled                              |
| Default encryption      | SSE-S3 (AES-256)                      |
| Bucket key              | Enabled                               |
| Object ownership        | Bucket owner enforced (ACLs disabled) |

**Bucket Policy:**

- **Role:** `stuttering-ai-model-read-role`
  - `s3:GetObject` → `stuttering-ai-logs/*`
  - `s3:ListBucket` → `stuttering-ai-logs`
- All other principals are implicitly denied.

**Encryption:** SSE-S3 (AES-256)

**Versioning:** Disabled

**Lifecycle Rules:**

| Rule                | Action                                        | Days |
| ------------------- | --------------------------------------------- | ---- |
| TransitionToGlacier | Move objects to S3 Glacier Flexible Retrieval | 90   |
| ExpireObjects       | Permanently delete objects                    | 365  |

> ✅ Logs are **immutable for 90 days**, then transitioned to Glacier, and automatically **deleted after 1 year**, controlling storage costs.

---

## Bucket Policies Application (AWS Console)

1. Sign in to **AWS Management Console** → navigate to **S3**.
2. Select the bucket.
3. Go to **Permissions → Bucket policy → Edit**.
4. Replace `ACCOUNT_ID` with your 12-digit AWS account ID.
5. Paste the corresponding policy JSON file from `docs/aws/bucket_policies/`.
6. Save changes.
7. Verify access by assuming the intended role and performing a test operation.

> **Important:** Incorrect policy ARNs can lock out all access except the root account. Use `ArnNotEquals` / `ArnNotLike` conditions carefully.
