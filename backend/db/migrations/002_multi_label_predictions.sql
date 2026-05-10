-- =============================================================================
-- Migration 002 — switch predictions to multi-label.
--
-- Replaces the single ``predictions.predicted_class`` column with a child
-- table ``prediction_classes`` that holds one row per detected class, with
-- its sigmoid confidence. This matches the v2.0 API contract (see
-- ``docs/api_contract.md``) and the multi-label model in
-- ``feature/multi-label-classification``.
--
-- Confidence values stored here are independent sigmoid probabilities — they
-- do NOT sum to 1.0 across the classes for a given prediction. Each class is
-- the output of a separate binary head.
--
-- ROLLBACK INSTRUCTIONS (run in a transaction):
--     BEGIN;
--     ALTER TABLE predictions
--         ADD COLUMN predicted_class stutterclass;
--     -- Backfill the legacy column with the highest-confidence class per row.
--     UPDATE predictions p
--     SET predicted_class = sub.class_label::stutterclass
--     FROM (
--         SELECT DISTINCT ON (prediction_id)
--                prediction_id, class_label
--         FROM   prediction_classes
--         ORDER  BY prediction_id, confidence DESC
--     ) sub
--     WHERE p.id = sub.prediction_id;
--     -- Drop the multi-label artifacts.
--     DROP INDEX IF EXISTS idx_prediction_classes_class_label;
--     DROP INDEX IF EXISTS idx_prediction_classes_prediction_id;
--     DROP TABLE IF EXISTS prediction_classes;
--     COMMIT;
-- =============================================================================

BEGIN;

-- 1. New child table: prediction_classes
CREATE TABLE IF NOT EXISTS prediction_classes (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Inline REFERENCES on prediction_id is the canonical FK; no separate
    -- named CONSTRAINT below to avoid a redundant duplicate FK in pg_constraint.
    prediction_id  UUID NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    class_label    TEXT NOT NULL,
    confidence     NUMERIC(6,5) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT chk_class_label CHECK (class_label = ANY(ARRAY[
        'fluent','blocks','interjections','prolongations',
        'part_word_repetition','phrase_repetition','word_repetition'
    ])),
    -- One row per (prediction, class) — never store the same class twice for
    -- the same prediction. Cheap idempotency for retried writers.
    CONSTRAINT uq_prediction_classes_prediction_id_class_label
        UNIQUE (prediction_id, class_label)
);

-- 2. Per-prediction lookup (the most common access pattern from
--    Laravel / FastAPI: "load all classes for prediction X").
CREATE INDEX IF NOT EXISTS idx_prediction_classes_prediction_id
    ON prediction_classes(prediction_id);

-- 3. Per-class lookup (analytics / "all sessions where blocks was detected").
CREATE INDEX IF NOT EXISTS idx_prediction_classes_class_label
    ON prediction_classes(class_label);

-- 4. Inline comment for future readers / DBAs.
COMMENT ON COLUMN prediction_classes.confidence IS
    'Independent sigmoid probability for this class. Values across rows for '
    'the same prediction_id do NOT sum to 1.0 — multi-label, not softmax.';

COMMENT ON TABLE prediction_classes IS
    'Per-class predictions for multi-label inference. One row per class '
    'whose sigmoid probability >= the server-side decision threshold at '
    'inference time. See docs/api_contract.md (v2.0).';

-- 5. Drop the legacy single-label column (its CHECK was via stutterclass enum).
ALTER TABLE predictions
    DROP COLUMN IF EXISTS predicted_class;

-- 6. Drop the now-unused index on the dropped column. Postgres normally drops
--    column-bound indexes implicitly with the column itself, but be explicit
--    so a re-run on a partially migrated DB doesn't error.
DROP INDEX IF EXISTS idx_predictions_predicted_class;

COMMIT;
