
-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";



-- Table: model_versions (created first — referenced by predictions FK)
CREATE TABLE IF NOT EXISTS model_versions (
    id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name       VARCHAR(100),
    model_version    VARCHAR(20)   NOT NULL UNIQUE,
    deployed_at      TIMESTAMPTZ,
    is_active        BOOLEAN       NOT NULL DEFAULT FALSE,
    artifact_path    TEXT,
    notes            TEXT
);



-- Table: predictions
CREATE TABLE IF NOT EXISTS predictions (
    id                   UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at           TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    audio_filename       VARCHAR(500),
    audio_duration_sec   FLOAT,
    predicted_class      VARCHAR(50)   NOT NULL,

    confidence_scores    JSONB         NOT NULL,

    model_version_id     UUID          NOT NULL,
    processing_time_ms   INTEGER,
    client_ip            INET,
    request_id           UUID          NOT NULL UNIQUE
);



-- Foreign Key: predictions → model_versions
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



-- Helper function: counts keys in a JSONB object.
CREATE OR REPLACE FUNCTION jsonb_key_count(j JSONB)
RETURNS INTEGER
LANGUAGE SQL
IMMUTABLE STRICT
AS $$
    SELECT count(*)::INTEGER FROM jsonb_object_keys(j);
$$;

-- Check constraint: confidence_scores must contain exactly
-- the 7 class keys and no others with FLOAT values in [0.0, 1.0]
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_confidence_scores_keys'
    ) THEN
        ALTER TABLE predictions
            ADD CONSTRAINT chk_confidence_scores_keys
            CHECK (
                jsonb_typeof(confidence_scores) = 'object'
                AND confidence_scores ?& ARRAY[
                    'fluent',
                    'blocks',
                    'interjections',
                    'prolongations',
                    'part_word_repetition',
                    'phrase_repetition',
                    'word_repetition'
                ]
                AND jsonb_key_count(confidence_scores) = 7
            );
    END IF;
END$$;


-- Indexes 
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at);

CREATE INDEX IF NOT EXISTS idx_predictions_predicted_class
    ON predictions (predicted_class);

CREATE INDEX IF NOT EXISTS idx_predictions_model_version_id
    ON predictions (model_version_id);