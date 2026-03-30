# IAM Roles and Least-Privilege S3 Access (WAE-06)

This document defines the IAM roles and policies used for S3 access in the project.
All policies follow least-privilege rules:

- no wildcard actions (`*` or `s3:*`)
- no wildcard resource ARNs (`arn:aws:s3:::*`)
- only required actions for each role
- path restrictions where applicable

## Role Summary

| Role | Purpose | Principal | Bucket | Allowed Actions | Resource Scope |
|---|---|---|---|---|---|
| `stuttering-ai-data-pipeline-role` | Ali data upload and read for raw dataset | IAM user profile | `stuttering-ai-data` | `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` | `raw/*` objects + bucket list with `raw` prefix |
| `stuttering-ai-model-read-role` | Backend/EC2 model read-only access | EC2 instance profile | `stuttering-ai-models` | `s3:GetObject`, `s3:ListBucket` | `models/*` objects + bucket list |
| `stuttering-ai-model-write-role` | Adan model artifact upload | IAM user profile | `stuttering-ai-models` | `s3:PutObject` | `models/*` objects only |

## Policy Files

- `docs/aws/iam_policies/stuttering_ai_data_pipeline_role_policy.json`
- `docs/aws/iam_policies/stuttering_ai_model_read_role_policy.json`
- `docs/aws/iam_policies/stuttering_ai_model_write_role_policy.json`

## Trust Relationships

Use these trust relationships when creating roles.

### Data pipeline role (Ali)

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

### Model write role (Adan)

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

### Model read role (Backend EC2)

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

## AWS CLI Setup Instructions

### Ali

1. Configure base profile:

```bash
aws configure --profile ali-base
```

2. Add assume-role profile in `~/.aws/config`:

```ini
[profile stuttering-ai-data-pipeline]
role_arn = arn:aws:iam::ACCOUNT_ID:role/stuttering-ai-data-pipeline-role
source_profile = ali-base
region = me-south-1
```

3. Test:

```bash
aws s3 ls s3://stuttering-ai-data/raw/ --profile stuttering-ai-data-pipeline
```

### Adan

1. Configure base profile:

```bash
aws configure --profile adan-base
```

2. Add assume-role profile in `~/.aws/config`:

```ini
[profile stuttering-ai-model-write]
role_arn = arn:aws:iam::ACCOUNT_ID:role/stuttering-ai-model-write-role
source_profile = adan-base
region = me-south-1
```

3. Test upload:

```bash
aws s3 cp model_inference.pt s3://stuttering-ai-models/models/model_inference.pt --profile stuttering-ai-model-write
```

### Backend (EC2)

- Attach `stuttering-ai-model-read-role` as instance profile to the EC2 instance.
- Do not set static credentials in code.
- Test from the instance:

```bash
aws s3 ls s3://stuttering-ai-models/models/
```

## Security Notes

- Replace `ACCOUNT_ID` before deployment.
- Rotate user access keys regularly.
- Never commit credentials.
- Keep role usage scoped to each task owner.
