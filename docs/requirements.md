
# Environment Matrix

This document defines the complete environment profiles for all project environments.

---

## 1. Local Development

**Runtime:** Python 3.10+

**Hardware:** CPU or CUDA (GPU optional)

### Environment Variables

| Variable                | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| `APP_ENV`             | `local`                                                               |
| `DATABASE_URL`        | `postgresql+asyncpg://<username>:<password>@localhost:5432/<db_name>` |
| `MODEL_ARTIFACT_PATH` | Local filesystem path to model artifact (e.g.,`./ai/models/artifact`) |
| `SECRET_KEY`          | Local development secret key                                            |
| `DEBUG`               | `true`                                                                |

### GPU / CPU

* CPU by default; CUDA supported if a local GPU is available and CUDA drivers are installed.
* PyTorch 2.x must be installed with the appropriate variant (`cpu` or `cu118` / `cu121`).

### Database Connection

* PostgreSQL running locally via Docker.
* Connect using `localhost:5432`.
* Managed with `docker-compose` or equivalent.

### Model Artifact Source

* Loaded from a local filesystem path defined by `MODEL_ARTIFACT_PATH`.

---



## 2. CI / Testing

**Runtime:** Python 3.10

**Hardware:** CPU only

**Platform:** GitHub Actions runner

### Environment Variables

| Variable                | Description                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| `APP_ENV`             | `ci`                                                                       |
| `DATABASE_URL`        | `postgresql+asyncpg://<username>:<password>@localhost:5432/<test_db_name>` |
| `MODEL_ARTIFACT_PATH` | Path to mock model artifact used in unit tests                               |
| `SECRET_KEY`          | Fixed test secret key                                                        |
| `DEBUG`               | `false`                                                                    |

### GPU / CPU

* CPU only. No GPU is available on standard GitHub Actions runners.
* PyTorch 2.x must be installed with the `cpu` variant.

### Database Connection

* PostgreSQL running as a GitHub Actions Docker service container.
* Connect using `localhost:5432` within the runner.

### Model Artifact Source

* A mock model object is used for unit tests, injected via test fixtures.
* `MODEL_ARTIFACT_PATH` may point to a minimal stub artifact for integration tests.

---


## 3. Production

**Runtime:** Python 3.10+ inside a Docker container

**Hardware:** GPU optional (EC2 / ECS instance type determines GPU availability)

**Platform:** Docker container deployed on AWS EC2 or ECS

### Environment Variables

| Variable                  | Description                                                             |
| ------------------------- | ----------------------------------------------------------------------- |
| `APP_ENV`               | `production`                                                          |
| `DATABASE_URL`          | `postgresql+asyncpg://<user>:<password>@<rds-endpoint>:5432/<dbname>` |
| `MODEL_ARTIFACT_BUCKET` | S3 bucket name where model artifacts are stored                         |
| `MODEL_ARTIFACT_KEY`    | S3 object key (path) for the model artifact                             |
| `SECRET_KEY`            | Production secret key (injected via secrets manager or environment)     |
| `DEBUG`                 | `false`                                                               |
| `AWS_REGION`            | AWS region (e.g.,`us-east-1`)                                         |
| `AWS_ACCESS_KEY_ID`     | AWS access key (or use IAM role attached to EC2/ECS)                    |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key (or use IAM role attached to EC2/ECS)                    |

### GPU / CPU

* GPU optional. The Docker image supports both CPU and GPU execution.
* If deployed on a GPU-enabled EC2 instance (e.g., `g4dn`, `p3`), the CUDA-enabled PyTorch variant must be used in the Docker image.
* If deployed on a CPU-only instance, the CPU PyTorch variant is used.

### Database Connection

* PostgreSQL on AWS RDS.
* Connect via the RDS endpoint over the private VPC network.
* SSL recommended for RDS connections in production.

### Model Artifact Source

* Model artifacts are loaded from AWS S3.
* `boto3` is used to download the artifact at container startup using `MODEL_ARTIFACT_BUCKET` and `MODEL_ARTIFACT_KEY`.
