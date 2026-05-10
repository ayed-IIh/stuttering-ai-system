-- =============================================================================
-- Stuttering-AI database schema (multi-label).
--
-- One ``predictions`` row per API call holds metadata and the per-class
-- sigmoid distribution (``all_scores`` JSONB). One ``prediction_classes``
-- child row per detected class holds (class_label, confidence) for the
-- subset that crossed the multi-label threshold.
--
-- IMPORTANT: confidence values in ``prediction_classes`` and the per-key
-- floats in ``all_scores`` are independent per-class sigmoid probabilities.
-- They do **NOT** sum to 1.0 across the seven classes — that's a softmax
-- invariant the new schema deliberately drops.
-- =============================================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Table: model_versions (created first, referenced by predictions FK)
CREATE TABLE IF NOT EXISTS model_versions (
    id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name       VARCHAR(100),
    model_version    VARCHAR(20)   NOT NULL UNIQUE,
    deployed_at      TIMESTAMPTZ,
    is_active        BOOLEAN       NOT NULL DEFAULT FALSE,
    artifact_path    TEXT,
    notes            TEXT
);

-- Table: predictions (parent row, one per API call)
CREATE TABLE IF NOT EXISTS predictions (
    id                   UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at           TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    audio_filename       VARCHAR(500),
    audio_duration_sec   FLOAT,
    all_scores           JSONB         NOT NULL,
    model_version_id     UUID          NOT NULL,
    processing_time_ms   INTEGER,
    client_ip            INET,
    request_id           UUID          NOT NULL UNIQUE
);

COMMENT ON COLUMN predictions.all_scores IS
    'Per-class sigmoid probabilities keyed by shared.labels.CLASS_LABELS. '
    'Independent per-class — do NOT sum to 1.0. Same value-set as the wire '
    'response field of the same name (see docs/api_contract.md v2.0).';

-- Foreign Key: predictions -> model_versions
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_predictions_model_version_id'
    ) THEN
        ALTER TABLE predictions
            ADD CONSTRAINT fk_predictions_model_version_id
            FOREIGN KEY (model_version_id)
            REFERENCES model_versions (id)
            ON DELETE RESTRICT;
    END IF;
END$$;

-- Table: prediction_classes (multi-label child of predictions)
-- One row per detected class for a given prediction (server-side threshold
-- applied). Empty predictions produce zero child rows.
CREATE TABLE IF NOT EXISTS prediction_classes (
    id             UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_id  UUID          NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    class_label    TEXT          NOT NULL,
    confidence     NUMERIC(6,5)  NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    created_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_prediction_classes_class_label CHECK (
        class_label = ANY(ARRAY[
            'fluent',
            'blocks',
            'interjections',
            'prolongations',
            'part_word_repetition',
            'phrase_repetition',
            'word_repetition'
        ])
    ),
    -- One row per (prediction, class) — never store the same class twice.
    CONSTRAINT uq_prediction_classes_prediction_id_class_label
        UNIQUE (prediction_id, class_label)
);

COMMENT ON TABLE prediction_classes IS
    'Per-class detections for multi-label inference. One row per class whose '
    'sigmoid probability >= the server-side decision threshold at inference '
    'time. See docs/api_contract.md (v2.0).';

COMMENT ON COLUMN prediction_classes.confidence IS
    'Independent sigmoid probability for this class. Values across rows for '
    'the same prediction_id do NOT sum to 1.0 — multi-label, not softmax.';

-- Indexes on predictions (kept after dropping idx_predictions_predicted_class)
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at);

CREATE INDEX IF NOT EXISTS idx_predictions_model_version_id
    ON predictions (model_version_id);

-- Indexes on prediction_classes
CREATE INDEX IF NOT EXISTS idx_prediction_classes_prediction_id
    ON prediction_classes (prediction_id);

CREATE INDEX IF NOT EXISTS idx_prediction_classes_class_label
    ON prediction_classes (class_label);
