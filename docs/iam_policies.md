# IAM Roles & Policies Documentation — Stuttering AI Project

**Project:** Stuttering AI

**Document:** IAM Roles & S3 Access Policies

**Version:** 1.0

**Last Updated:** 2026-03-25

**Reference:** WAE-06

---

## Table of Contents

1. [Design Principles](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#design-principles)
2. [Role Summary](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#role-summary)
3. [Role 1: stuttering-ai-data-pipeline-role](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#role-1-stuttering-ai-data-pipeline-role)
4. [Role 2: stuttering-ai-model-read-role](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#role-2-stuttering-ai-model-read-role)
5. [Role 3: stuttering-ai-model-write-role](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#role-3-stuttering-ai-model-write-role)
6. [How to Create Roles &amp; Attach Policies (AWS Console)](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#how-to-create-roles--attach-policies-aws-console)
7. [AWS CLI Profile Setup — Per Team Member](https://claude.ai/chat/8159afa9-e2a8-4756-a004-b7e2660c199d#aws-cli-profile-setup--per-team-member)

---

## Design Principles

All IAM policies in this project follow strict  **least-privilege** :

* No wildcard actions (`s3:*` or `*`) are permitted anywhere.
* No wildcard resource ARNs (`arn:aws:s3:::*`) are permitted anywhere.
* Each role grants the minimum set of S3 actions required for its specific task.
* Path-level restrictions (via resource ARN suffixes) confine each role to the exact S3 prefix it needs.
* No cross-bucket permissions are granted; each role touches exactly one bucket.

---

## Role Summary

| Role Name                            | Assigned To                      | Bucket                   | Actions                                    | Path Restriction        |
| ------------------------------------ | -------------------------------- | ------------------------ | ------------------------------------------ | ----------------------- |
| `stuttering-ai-data-pipeline-role` | Ali (IAM user / access key)      | `stuttering-ai-data`   | `PutObject`,`GetObject`,`ListBucket` | `raw/*`only           |
| `stuttering-ai-model-read-role`    | Backend / EC2 (instance profile) | `stuttering-ai-models` | `GetObject`,`ListBucket`               | All objects (read-only) |
| `stuttering-ai-model-write-role`   | Adan (IAM user / access key)     | `stuttering-ai-models` | `PutObject`                              | `models/*`only        |

---

## Role 1: `stuttering-ai-data-pipeline-role`

### Purpose

Allows Ali to upload raw datasets to the `stuttering-ai-data` bucket and retrieve them for preprocessing. Access is restricted to the `raw/` prefix; no access to `processed/` or any other bucket.

### AWS Principal

**IAM User:** `ali` (or the IAM user account assigned to Ali)

**Access method:** Access key exported to Ali's local AWS CLI profile.

### Permissions

| Action            | Resource ARN                              | Condition                                         |
| ----------------- | ----------------------------------------- | ------------------------------------------------- |
| `s3:PutObject`  | `arn:aws:s3:::stuttering-ai-data/raw/*` | None                                              |
| `s3:GetObject`  | `arn:aws:s3:::stuttering-ai-data/raw/*` | None                                              |
| `s3:ListBucket` | `arn:aws:s3:::stuttering-ai-data`       | `s3:prefix`must start with `raw/`or `raw/*` |

### Policy File

`docs/aws/iam_policies/stuttering-ai-data-pipeline-role-policy.json`

### How to Assume

Ali assumes this role via an **access key** exported in his AWS CLI profile. he does **not** use an instance profile. The role's trust policy must allow her IAM user ARN to perform `sts:AssumeRole`.

**Trust policy (attach to the role in IAM Console → Trust relationships):**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::ACCOUNT_ID:user/ali"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

---

## Role 2: `stuttering-ai-model-read-role`

### Purpose

Grants the backend application (running on EC2) read-only access to all model artifacts in `stuttering-ai-models`. The backend can list and download any object but cannot upload, delete, or modify anything.

### AWS Principal

**EC2 Instance Profile:** Attached directly to the EC2 instance(s) running the backend service.

**Access method:** EC2 instance profile — no access keys are issued or required. AWS SDKs automatically retrieve temporary credentials from the instance metadata service (IMDS).

### Permissions

| Action            | Resource ARN                            | Condition |
| ----------------- | --------------------------------------- | --------- |
| `s3:GetObject`  | `arn:aws:s3:::stuttering-ai-models/*` | None      |
| `s3:ListBucket` | `arn:aws:s3:::stuttering-ai-models`   | None      |

### Policy File

`docs/aws/iam_policies/stuttering-ai-model-read-role-policy.json`

### How to Assume

EC2 assumes this role automatically via its  **instance profile** . No `sts:AssumeRole` call is needed by application code — the AWS SDK resolves credentials from IMDS at `http://169.254.169.254`.

**Trust policy (attach to the role in IAM Console → Trust relationships):**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**How to attach to EC2 (AWS Console):**

1. IAM Console → **Roles** → select `stuttering-ai-model-read-role` → confirm **Instance Profile** is listed.
2. EC2 Console → select the instance → **Actions** → **Security** → **Modify IAM role** → select `stuttering-ai-model-read-role` →  **Update IAM role** .

---

## Role 3: `stuttering-ai-model-write-role`

### Purpose

Allows Adan to upload trained model artifacts to `stuttering-ai-models` after a training run completes. Write access is restricted to the `models/` prefix only. Adan cannot read, list, or delete objects.

### AWS Principal

**IAM User:** `adan` (or the IAM user account assigned to Adan)

**Access method:** Access key exported to Adan's local AWS CLI profile.

### Permissions

| Action           | Resource ARN                                   | Condition |
| ---------------- | ---------------------------------------------- | --------- |
| `s3:PutObject` | `arn:aws:s3:::stuttering-ai-models/models/*` | None      |

### Policy File

`docs/aws/iam_policies/stuttering-ai-model-write-role-policy.json`

### How to Assume

Adan assumes this role via an **access key** exported in his AWS CLI profile. The role's trust policy must allow his IAM user ARN.

**Trust policy (attach to the role in IAM Console → Trust relationships):**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::ACCOUNT_ID:user/adan"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

---

## How to Create Roles & Attach Policies (AWS Console)

Follow these steps for each of the three roles. The steps are the same for all three; substitute the role name and policy file as appropriate.

### Step 1 — Create the IAM Policy

1. Sign in to the **AWS Console** → navigate to **IAM** → **Policies** →  **Create policy** .
2. Select the **JSON** tab.
3. Replace `ACCOUNT_ID` in the JSON file with your 12-digit AWS Account ID.
4. Paste the contents of the corresponding policy JSON file (from `docs/aws/iam_policies/`).
5. Click  **Next** .
6. **Policy name:** use the same name as the role with `-policy` suffix, e.g. `stuttering-ai-data-pipeline-role-policy`.
7. Click  **Create policy** .

### Step 2 — Create the IAM Role

1. IAM Console → **Roles** →  **Create role** .
2. **Trusted entity type:**
   * For `stuttering-ai-data-pipeline-role` and `stuttering-ai-model-write-role`: select **AWS account** → **This account** (you will scope to the specific IAM user next).
   * For `stuttering-ai-model-read-role`: select **AWS service** →  **EC2** .
3. Click  **Next** .
4. Search for and attach the policy you created in Step 1.
5. Click  **Next** .
6. **Role name:** enter the exact role name (e.g., `stuttering-ai-data-pipeline-role`).
7. Click  **Create role** .

### Step 3 — Scope the Trust Policy to the Specific IAM User (user-assumed roles only)

> Skip this step for `stuttering-ai-model-read-role` (EC2 instance profile).

1. IAM Console → **Roles** → click the role name.
2. **Trust relationships** tab →  **Edit trust policy** .
3. Replace the default trust policy with the role-specific trust policy shown in the relevant section above (substituting `ACCOUNT_ID` and the username).
4. Click  **Update policy** .

### Step 4 — Generate Access Keys (user-assumed roles only)

> Skip this step for `stuttering-ai-model-read-role` (EC2 instance profile).

Each team member needs an IAM user with permission to call `sts:AssumeRole` on their assigned role.

1. IAM Console → **Users** → select the user (e.g., `ali`).
2. **Security credentials** tab → **Access keys** →  **Create access key** .
3. Select **CLI** as the use case.
4. Download or copy the **Access Key ID** and **Secret Access Key** — these are shown only once.
5. Share securely with the team member (use a password manager or encrypted channel — never email or Slack).

---

## AWS CLI Profile Setup — Per Team Member

### Prerequisites

* AWS CLI v2 installed: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
* Access Key ID and Secret Access Key provided by the project admin.

---

### Ali — `stuttering-ai-data-pipeline-role`

Ali's CLI is configured with her IAM user credentials. he then assumes `stuttering-ai-data-pipeline-role` via a named profile so all pipeline commands automatically use the scoped role.

**Step 1 — Configure the base IAM user credentials:**

Open **AWS Console** → **IAM** → **Users** → `ali` → **Security credentials** → copy Access Key ID and Secret Access Key.

Then configure on Ali's machine:

In a terminal, run:

```
aws configure --profile ali-base
```

Enter:

* AWS Access Key ID: `<ali's access key id>`
* AWS Secret Access Key: `<ali's secret access key>`
* Default region name: `us-east-1` (or your project region)
* Default output format: `json`

**Step 2 — Add the role-assumption profile:**

Open (or create) `~/.aws/config` in a text editor and append:

```ini
[profile stuttering-ai-data-pipeline]
role_arn = arn:aws:iam::ACCOUNT_ID:role/stuttering-ai-data-pipeline-role
source_profile = ali-base
region = me-south-1
```

**Step 3 — Test:**

```
aws s3 ls s3://stuttering-ai-data/raw/ --profile stuttering-ai-data-pipeline
```

Expected result: lists contents of `raw/` (empty on first run). No `AccessDenied` error means the role assumption succeeded.

---

### Adan — `stuttering-ai-model-write-role`

Identical flow to Ali but using Adan's IAM user credentials and the write role.

**Step 1 — Configure the base IAM user credentials:**

Open **AWS Console** → **IAM** → **Users** → `adan` → **Security credentials** → copy credentials.

Run:

```
aws configure --profile adan-base
```

Enter Adan's Access Key ID, Secret Access Key, region, and output format.

**Step 2 — Add the role-assumption profile:**

Append to `~/.aws/config`:

```ini
[profile stuttering-ai-model-write]
role_arn = arn:aws:iam::ACCOUNT_ID:role/stuttering-ai-model-write-role
source_profile = adan-base
region = us-east-1
```

**Step 3 — Test:**

Create a small test file and upload it:

```
echo "test" > test-model.txt
aws s3 cp test-model.txt s3://stuttering-ai-models/models/test-model.txt --profile stuttering-ai-model-write
```

Expected result: upload succeeds. Then clean up:

```
aws s3 rm s3://stuttering-ai-models/models/test-model.txt --profile stuttering-ai-model-write
```

> Note: `s3 rm` will fail with `AccessDenied` because the write role does not have `s3:DeleteObject`. That is expected and correct — delete the test file manually via the AWS Console (S3 → bucket → object → Delete) if needed.

---

### Backend / EC2 — `stuttering-ai-model-read-role`

The backend does **not** use an access key or a named CLI profile. The role is attached to the EC2 instance as an  **instance profile** .

**No CLI configuration is needed on the EC2 instance.** The AWS SDK (Python boto3, Node.js SDK, etc.) automatically resolves credentials from the instance metadata service (IMDS) when no explicit credentials are provided.

To verify from inside the EC2 instance:

```
aws s3 ls s3://stuttering-ai-models/
```

Expected result: lists the `models/` prefix. No profile flag is needed — the instance profile provides credentials automatically.

If the command returns `Unable to locate credentials`, confirm the instance profile is attached: EC2 Console → instance → **Security** tab → **IAM Role** should show `stuttering-ai-model-read-role`.

---

## Notes

* Replace `ACCOUNT_ID` in all ARNs with the actual 12-digit AWS Account ID before use.
* Access keys should be rotated every 90 days per AWS best practice. Use IAM Console → user → Security credentials → Create access key, then update `~/.aws/credentials` and deactivate the old key.
* Never commit access keys to version control. Add `~/.aws/` to `.gitignore` as a safeguard.
* For local development on EC2-like environments, consider using AWS SSO or `aws-vault` instead of long-lived access keys.
