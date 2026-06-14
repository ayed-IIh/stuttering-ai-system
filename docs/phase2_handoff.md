# Phase 2 Backend Handoff

Technical handoff for the current FastAPI + PostgreSQL backend and the planned Phase 2 roadmap.

---

## 1) Local Setup

### Prerequisites

- **Python** 3.10 or newer (see project `README.md`)
- **pip** and a virtual environment (recommended)
- **Dependencies:** `pip install -r requirements.txt -r requirements-dev.txt`
- **Optional:** Docker Desktop for the compose stack (`postgres`, backend, pgAdmin)

### Environment variables (`.env`)

Copy `.env.example` to `.env` and configure at least the following.

**PostgreSQL (required for `backend.db.database` import and runtime)** — the async engine URL is built from these; missing values cause startup failure when the DB package loads:

- `POSTGRES_HOST` — e.g. `localhost` (host) or `postgres` (Docker service name)
- `POSTGRES_PORT` — e.g. `5432`
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`

**Application / inference (`backend.app.config.Settings`, via `.env`)** — see comments in `.env.example`:

- `MODEL_PATH` — directory with `model_inference.pt` and `config.json` (or artifact path your `ModelService` expects)
- `MODEL_SOURCE` — `local` or `s3` (S3 path may be stubbed until extended)
- `DEVICE` — `cpu` or `cuda`
- `MAX_AUDIO_SIZE_MB` — max upload size for prediction
- `ALLOWED_ORIGINS` — CORS (comma-separated or JSON list); production rejects `*`
- `PRODUCTION_MODE` — `true` / `false`
- `DB_URL` — sync-style URL used by settings (ORM/async URL for workers may still use `POSTGRES_*` composition in `database.py`)
- `SERVICE_VERSION` — exposed in OpenAPI and health
- `LOG_LEVEL` — `DEBUG` | `INFO` | `WARNING` | `ERROR` | `CRITICAL`

Optional compose-only variables (`APP_ENV`, `DEBUG`, pgAdmin, etc.) are documented in `.env.example`.

### Startup command

From the repository root (virtualenv activated):

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Docker (full stack):** see `README.md` — e.g. `docker compose -f docker-compose.dev.yml up --build -d`, then `http://127.0.0.1:8000`.

### Expected output

- Uvicorn process listening on the configured host/port
- Structured JSON logs on stdout (`structlog`): e.g. `app_startup`, then successful model load logs such as `model_ready` with `loaded=true` (exact event names depend on `ModelService` implementation)
- **`GET /health`** returns **200** with JSON including service health and model readiness, e.g. `status`, `model_loaded`, `version`, and timing fields such as `uptime_seconds` (see OpenAPI `/docs` once the app runs)
- **`GET /docs`** — Swagger UI for the mounted routes

---

## 2) Model artifact swap (no code changes)

1. **Build or export** a new inference bundle: directory containing at least `model_inference.pt` and `config.json` (paths/names must match what `ModelService` loads today).
2. **Validate `config.json`** includes what the loader expects (e.g. Hugging Face `model_name` / processor id, class label list or mapping, optional `model_version`, `sample_rate`, `max_samples`).
3. **Deploy files** to the server (example: `/opt/stuttering/models/2026-03-29/`).
4. **Point the app at the new directory** by updating **only** `.env`:
   - `MODEL_PATH=<new_directory>`
   - Optionally bump `SERVICE_VERSION` for traceability.
5. **Restart** the API process (or recreate the backend container) so lifespan reload runs.
6. **Smoke test:** `GET /health` → 200; `POST /api/v1/predict` with a known-good WAV → response matches the public prediction schema; check logs for successful load and no `model_load_failed`.

No Python source edits are required if the artifact layout and `config.json` keys stay compatible with the existing loader.

---

## 3) Known limitations

- **Single file only** — no batch prediction API in the current design.
- **Synchronous inference** — the request is held until inference (and mandatory preprocessing) finishes.
- **No streaming** — no chunked or progressive prediction response.
- **No model versioning HTTP API** — no `GET /models` or equivalent registry endpoint yet.
- **No async job queue** — no Celery/RQ/FastAPI background offload for long-running inference.

---

## 4) Planned Phase 2 improvements

- **Async inference / queue** — Celery (or similar) workers, or lighter **FastAPI `BackgroundTasks`** for decoupled work; add task id, status polling, timeouts, and retries.
- **Model versioning endpoint** — e.g. **`GET /models`**: active version, available artifacts, checksums/metadata.
- **Batch prediction** — multi-file upload with per-item success/failure and stable `request_id` per sub-request.
- **Mobile app authentication** — validate tokens (e.g. JWT) in middleware; tie predictions to user/session context and enforce quotas if needed.

---

## 5) API endpoint reference summary

| Method | Path | One-line description | Response schema (summary) |
|--------|------|----------------------|---------------------------|
| `GET` | `/health` | Liveness/readiness: API up and model load state | JSON: `status`, `model_loaded`, `version`, `uptime_seconds` (and related fields as implemented) |
| `GET` | `/api/v1/classes` | Returns canonical class taxonomy for the 7-way classifier | JSON: `classes`, `label_to_id`, `id_to_label` (aligned with `shared.labels`) |
| `POST` | `/api/v1/predict` | Accepts one WAV (`multipart/form-data`), runs inference, returns scores | JSON: `predicted_class`, `confidence_scores` (per-class probabilities), `processing_time_ms`, `model_version`, `request_id` |

*Detailed request constraints (max size, MIME, form field names) live in `docs/api_contract.md` when that branch is merged, or in OpenAPI `/docs`.*

---

## 6) Open risks and questions for Phase 2

- **Model cold start** — time and memory spike on process start or first GPU allocation after deploy.
- **Concurrency** — memory pressure when many parallel `POST /api/v1/predict` requests hit a large Wav2Vec2-style model on CPU or shared GPU.
- **Mobile audio diversity** — varying WAV headers, sample rates, mono/stereo, and truncated uploads causing validation vs. model edge cases.
- **Operational clarity** — whether `DATABASE_URL` in `.env.example` and `POSTGRES_*` composition in `database.py` should be unified to a single source of truth.
- **Persistence volume** — growth and retention policy for prediction/audit rows if logging every request.

---

Reviewed by Wael before merging.
