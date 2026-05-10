# Stuttering Inference Service — REST API Contract

**Document version:** 2.0
**Status:** Approved — supersedes the draft v1.0 single-label contract.
**Related work:** `feature/multi-label-classification` switches loss to BCE,
decode to sigmoid + threshold, and response shape to multi-label.

This document is the authoritative contract for the REST endpoints exposed by
the FastAPI inference service. The implementation in `backend/api/routes.py`
must match this document; any drift is a bug.

---

## 1. Conventions

- **Base path:** `/api/v1/` (router mounted at this prefix by `backend/app/main.py`).
- **Content-Type:** all success and error responses use `application/json`.
- **Character encoding:** UTF-8.
- **Class names** appearing in any response field always come from
  `shared.labels.CLASS_LABELS`. They are the 7-element lowercase taxonomy:
  ```text
  fluent, blocks, interjections, prolongations,
  part_word_repetition, phrase_repetition, word_repetition
  ```
  Do not capitalize, alias, or rename them on the wire.

---

## 2. Endpoints

### 2.1 POST /api/v1/predict

Submit a WAV file for **multi-label** stuttering classification. The model
runs an independent binary classifier per class; any subset (including the
empty set) may be returned.

#### Request

| Aspect | Specification |
|---|---|
| Method | `POST` |
| Content-Type | `multipart/form-data` |
| Body | Form data with one field: `audio_file` |

| Field | Type | Required | Description |
|---|---|---|---|
| `audio_file` | binary (file) | **Yes** | WAV. MIME must be `audio/wav` or `audio/x-wav`. Magic bytes must start with `RIFF`. Max size from `MAX_AUDIO_SIZE_MB` (default 10 MB). |

There is **no** `sample_rate_hint` field (removed in v2.0). The server resamples
internally to `audio_loader.TARGET_SAMPLES` / `target_sr`.

#### Success response (200 OK)

```json
{
  "predicted_classes": [
    {"class": "blocks",        "confidence": 0.87},
    {"class": "prolongations", "confidence": 0.64}
  ],
  "all_scores": {
    "fluent":               0.04,
    "blocks":               0.87,
    "interjections":        0.05,
    "prolongations":        0.64,
    "part_word_repetition": 0.11,
    "phrase_repetition":    0.03,
    "word_repetition":      0.06
  },
  "threshold": 0.5,
  "processing_time_ms": 145,
  "model_version": "1.0.0",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Description |
|---|---|---|
| `predicted_classes` | array of `{class, confidence}` | Every class whose sigmoid probability is `>= threshold`. **May be empty.** Sorted by descending confidence. |
| `predicted_classes[].class` | string | One of `CLASS_LABELS`. |
| `predicted_classes[].confidence` | number ∈ [0, 1] | Sigmoid probability for this class. |
| `all_scores` | object | Always exactly 7 keys (one per `CLASS_LABELS` entry). **Independent sigmoid probabilities — values do NOT sum to 1.0.** |
| `threshold` | number ∈ [0, 1] | The decision threshold the server applied to produce `predicted_classes`. Sourced from `Settings.MULTI_LABEL_THRESHOLD`. |
| `processing_time_ms` | integer ≥ 0 | Wall-clock milliseconds spent inside `ModelService.predict`. |
| `model_version` | string | Identifier of the loaded model artifact. |
| `request_id` | string (UUID v4) | Server-generated per request, returned for correlation. |

#### Error responses

All errors use the [Standard error envelope](#4-standard-error-envelope).

| Status | error_code | Triggered by |
|---|---|---|
| 400 | `INVALID_REQUEST` | Malformed form data, invalid WAV magic bytes. |
| 413 | `FILE_TOO_LARGE` | Uploaded body or declared content-length > `MAX_AUDIO_SIZE_MB`. |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | MIME type not in `{audio/wav, audio/x-wav}`. |
| 422 | `UNPROCESSABLE_AUDIO` | File is valid WAV by header but cannot be decoded/preprocessed (e.g. corrupt frames, missing audio data). Also returned when the required `audio_file` field is missing (FastAPI form validation). |
| 500 | `MODEL_ERROR` | `PredictionError` raised inside the inference path. |
| 503 | `SERVICE_UNAVAILABLE` | `ModelNotLoadedError` (model has not finished loading or was not initialized). |

---

### 2.2 GET /api/v1/health

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

503 with `SERVICE_UNAVAILABLE` is returned only when `request.app.state.model_service` is `None` (service not constructed).

### 2.3 GET /api/v1/classes

Returns the taxonomy that the response classes are drawn from.

```json
{
  "classes": ["fluent", "blocks", "interjections", "prolongations",
              "part_word_repetition", "phrase_repetition", "word_repetition"],
  "label_to_id": {
    "fluent": 0, "blocks": 1, "interjections": 2, "prolongations": 3,
    "part_word_repetition": 4, "phrase_repetition": 5, "word_repetition": 6
  },
  "id_to_label": {"0": "fluent", "1": "blocks", "...": "..."}
}
```

---

## 3. Threshold Semantics

`threshold` controls which classes appear in `predicted_classes`:

- A class is **included** when `sigmoid(logit) >= threshold`.
- Equality counts as inclusion (so `threshold = 0.5` with `sigmoid = 0.5` includes the class).
- `threshold` is a server-side setting (`MULTI_LABEL_THRESHOLD` in
  `backend/app/config.py`), not a request parameter. Clients receive the value
  the server applied so they can re-filter `all_scores` locally if desired.
- The default is `0.5`. Per-class threshold tuning is expected at evaluation
  time; when adopted, the wire field will become a `dict[str, float]` (one per
  class) in a future minor version. For v2.0 it is a single float.

---

## 4. Standard error envelope

```json
{
  "error_code": "FILE_TOO_LARGE",
  "message": "Audio file exceeds maximum allowed size",
  "detail": "Declared size exceeds max: 10485760 bytes"
}
```

In production mode (`Settings.PRODUCTION_MODE == True`) the `detail` field is
omitted to avoid leaking internal state. In development mode it carries the
human-readable cause.

---

## 5. Empty Result

`predicted_classes` MAY be the empty list. This happens when the model's
sigmoid output for every class is strictly below `threshold`.

Recommended client behavior:

- **UI**: show a neutral message like *"No stuttering detected above the
  confidence threshold."* Do not silently fall back to "fluent" — the
  distinction matters clinically.
- **Database**: store the empty result as a session with zero rows in the
  `prediction_classes` child table (see DB migration 002 — Step 9 work). Do
  not coerce it into a `predicted_class="fluent"` row.

An empty result is NOT an error and does NOT cause a non-200 status code.

---

## 6. Important properties (read these)

1. **`all_scores` is NOT a probability distribution.** Each value is an
   independent binary sigmoid. They do not sum to 1.0, and the maximum may not
   be the most "diagnostic" class — multiple may exceed the threshold simultaneously.

2. **Class names are sourced from `shared/labels.py::CLASS_LABELS` everywhere.**
   The server validates this at response build time (`PredictionResponse.all_scores`
   field validator). If you see a different name or a missing key in any response,
   that is a server bug — file an issue.

3. **No client-supplied `sample_rate_hint`.** Sample rate is determined from the
   uploaded WAV header and resampled internally.

4. **`/api/v1/health` is the only health endpoint** clients should call.
   `backend/app/main.py` also registers `/health` at the root for backwards
   compatibility; this is undocumented and may be removed.

---

## 7. Approval checklist

- [x] Reviewed by Adan (model / output format alignment).
- [x] Reviewed by Wael (DB schema alignment — see migration 002).
- [x] Reviewed by Saddouq (API surface, error matrix).
- [x] Implementation in `backend/api/routes.py` matches this document.
- [x] Tests in `backend/tests/test_routes.py` assert this contract.

**End of contract v2.0.**
