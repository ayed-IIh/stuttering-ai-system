-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";


-- Table: model_versions
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(20) UNIQUE,
    deployed_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT FALSE,
    artifact_path TEXT,
    notes TEXT
);


-- Table: predictions
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    audio_filename VARCHAR(500),
    audio_duration_sec FLOAT,
    predicted_class VARCHAR(50) NOT NULL,
    confidence_score JSONB NOT NULL,
    model_version_id UUID NOT NULL,
    processing_time_ms INTEGER,
    client_ip INET,
    request_id UUID UNIQUE NOT NULL
);


-- Foreign Key Constraint
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_predictions_model_version_id'
    ) THEN
        ALTER TABLE predictions
        ADD CONSTRAINT fk_predictions_model_version_id
        FOREIGN KEY (model_version_id)
        REFERENCES model_versions(id)
        ON DELETE RESTRICT;
    END IF;
END$$;


-- Indexes
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
ON predictions(created_at);

CREATE INDEX IF NOT EXISTS idx_predictions_predicted_class
ON predictions(predicted_class);

CREATE INDEX IF NOT EXISTS idx_predictions_model_version_id
ON predictions(model_version_id);