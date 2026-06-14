# Stuttering AI — Deployment Guide

This service classifies an audio clip into one of 7 stuttering classes
(`fluent, blocks, interjections, prolongations, part_word_repetition,
phrase_repetition, word_repetition`). It is **single-label** internally
(softmax + argmax) but exposes a **multi-label-shaped contract** so the mobile
stack (Laravel `mobeen-mob-api` + Flutter `mobeen-mob-app`) works unchanged.

---

## 1. Architecture

```
 Flutter app ──► Laravel mobeen-mob-api ──► AI service (this repo)
   (therapist)        (server-to-server)        FastAPI :8000
                                                 ├─ POST /api/v1/predict
                                                 └─ POST /api/v1/feedback  (HITL)
```

The mobile API calls the AI **server-to-server** (not from the browser), so
CORS is not on the integration path.

---

## 2. API contract (what the mobile requires)

### `POST /api/v1/predict`
- **Request:** one audio source, either —
  - `multipart/form-data` with file field **`audio_file`** (WAV, 16 kHz mono), or
  - form field **`s3_key`** (when `STORAGE_BACKEND=s3`): the mobile uploads the
    clip to `s3://<S3_BUCKET>/<key>` and sends the key; the AI downloads it.
  - Exactly one is required (else `422 MISSING_AUDIO`).
- **Response (200):**
```json
{
  "predicted_class": "blocks",
  "confidence_scores": { "fluent": 0.18, "blocks": 0.62, "...": 0.0 },
  "predicted_classes": [ { "class": "blocks", "confidence": 0.62 } ],
  "all_scores": { "fluent": 0.18, "blocks": 0.62, "...": 0.0 },
  "threshold": 0.62,
  "processing_time_ms": 134,
  "model_version": "v3.0",
  "request_id": "uuid"
}
```
- `predicted_classes`, `all_scores`, `threshold` are the fields the mobile
  **requires** (it throws if missing). For single-label output,
  `predicted_classes` holds exactly one entry (the argmax winner) and
  `threshold` equals the winner's confidence, so the mobile's history-recompute
  (`all_scores >= threshold`) also yields just the winner.
- Scores are **0–1**; the mobile auto-scales them to 0–100 for display.
- `predicted_class` + `confidence_scores` are retained for the DB layer and
  direct (non-mobile) consumers.

### `POST /api/v1/feedback`  (Human-in-the-loop)
- Called by the mobile **only when a therapist edits the diagnosis**
  (fire-and-forget, 10 s timeout, audio ≤ 5 MB).
- **Request (JSON):**
```json
{
  "audio_base64": "<base64 WAV>",
  "correct_labels": ["blocks", "prolongations"],
  "original_prediction": ["fluent"],
  "model_version": "v3.0"
}
```
- **Response (200):** `{ "status": "accepted", "feedback_id": "...", "stored_count": 12 }`
- Stores the WAV + labels under `FEEDBACK_DIR` (`audio/<id>.wav` +
  `feedback.jsonl`). This is the dataset that grows the rare classes over time;
  feed it back into training to lift accuracy.

### Other endpoints
`GET /api/v1/health`, `GET /api/v1/classes`, `GET /health`.

---

## 3. Run with Docker (recommended)

```bash
cp .env.example .env        # then edit (see §4)
docker compose up --build
```

The image is **self-contained**: it bakes the v3.0 model (~360 MB) and
pre-bakes the HuggingFace processor cache, so it loads **offline** with no
network call at startup.

> **Build prerequisite:** `exports/v3.0/{model_inference.pt, config.json}` must
> exist in the build context. The model is git-ignored (kept locally), so build
> on a machine that has it, or copy it in before `docker build`.

Health check: `curl http://localhost:8000/health`.

---

## 4. Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL_PATH` | `/app/exports/v3.0` (in image) | Dir with `model_inference.pt` + `config.json` |
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `SERVICE_VERSION` | `v3.0` | Reported in `/health` + stored with predictions |
| `FEEDBACK_DIR` | `/app/feedback_data` (volume) | HITL corrections store |
| `MAX_AUDIO_SIZE_MB` | `10` | `/predict` upload cap |
| `PRODUCTION_MODE` | `false` | **Set `true` in prod** — strict model load, rejects CORS `*`, hides error details |
| `ALLOWED_ORIGINS` | localhost | CORS (only needed if a browser calls the AI directly) |
| `POSTGRES_*` | unset | **Optional.** If unset, DB is disabled and predictions just aren't logged (service still serves predict + feedback) |
| `DB_ECHO` | `false` | Verbose SQL logging |
| `STORAGE_BACKEND` | `local` | `local` (FEEDBACK_DIR) or `s3` (audio + corpus in S3) |
| `S3_BUCKET` | — | Required when `STORAGE_BACKEND=s3` |
| `S3_REGION` / `S3_ENDPOINT_URL` | — | AWS region / endpoint override (S3-compatible or mock) |
| `S3_PREDICT_PREFIX` / `S3_FEEDBACK_PREFIX` | `predict-audio/` / `feedback/` | Key prefixes |
| `MAX_FEEDBACK_AUDIO_MB` | `5` | `/feedback` decoded-WAV cap |

**Postgres is optional.** A missing `POSTGRES_*` no longer blocks boot — the DB
is used only to log prediction rows.

**S3 (`STORAGE_BACKEND=s3`).** AWS credentials come from the standard chain
(env vars / IAM role / `~/.aws`). With S3 on: `/predict` accepts an `s3_key`
(the AI downloads the clip), and `/feedback` stores each correction as
`<S3_FEEDBACK_PREFIX><id>.wav` + `<id>.json` — a horizontally-scalable corpus
(no shared local file). Local mode stays the zero-dependency default.

---

## 4b. Automated retraining (HITL)

`scripts/retrain.py` (scheduled via `ops/cron/stuttering-ai-retrain.cron`,
weekly) closes the loop: it pulls the accumulated corrections (S3 or local),
augments the training manifest, retrains from the current production checkpoint,
evaluates on the held-out test set, and **promotes the new model only if its
macro-F1 improves**. It self-skips when fewer than `--min-new` corrections have
arrived since the last run, so weekly runs are cheap.

> Retraining is **batch, not real-time** (real-time per-correction training
> causes catastrophic forgetting). Requires the training deps + the original
> dataset on the host (GPU recommended). The corpus reader + manifest builder
> are unit-tested; the train/evaluate steps shell out to the existing
> `ai/training/train_with_init.py` and `ai/evaluation/evaluate.py`.

---

## 5. Connecting the mobile stack

The mobile API's defaults point at:
```
AI_PREDICT_URL=http://ai-service:8000/api/v1/predict
AI_FEEDBACK_URL=http://ai-service:8000/api/v1/feedback
```
This compose exposes the backend under the network alias **`ai-service`**, so
when both stacks share a Docker network the default URLs resolve as-is. If you
deploy them separately, set `AI_PREDICT_URL` / `AI_FEEDBACK_URL` on the mobile
side to wherever this service is reachable.

---

## 6. Status / what's left

**Done & verified**
- `/predict` emits the multi-label-shaped contract the mobile requires, and
  accepts either a multipart upload or an `s3_key`.
- `/feedback` endpoint + HITL store (local JSONL **or** S3 per-object corpus).
- S3 storage layer (audio + corpus), unit-tested with `moto`.
- Automated retraining script + weekly cron (corpus → retrain → eval → promote
  if better); data layer unit-tested.
- Boots without Postgres; DB optional. Non-root container; offline model load.
- 28 backend/retrain tests pass.

**Not done yet (future)**
- **Mobile-side S3 upload for `/predict`:** the AI accepts `s3_key`, but the
  mobile-API (Laravel) must implement uploading the clip to S3 + sending the
  key. Until then the multipart path keeps working.
- **`_persist_prediction` runs inline in `/predict`** — make it a background
  task so a slow DB never inflates predict latency.
- **Feedback store (local mode) is single-host** (in-process lock + `fsync`);
  the **S3 backend** removes this limit for horizontal scaling. No retention
  policy yet — monitor the corpus size.
- Migrate the remaining Pydantic v1 `@validator`/`.dict()` in
  `backend/db/schemas.py` + `crud.py` to v2; add the strict
  `chk_confidence_scores_keys` constraint to the ORM (matches the SQL migration).

**Hardening already applied:** non-root container user (`appuser`), offline
model load (baked HF cache), DB-optional boot, WAV+label validation on
`/feedback`, blocking feedback/S3 I/O offloaded to the threadpool, taxonomy
sourced from `shared.labels` (no duplication), CI no longer masks test
failures, and a parameterized `MODEL_VERSION` build arg.
