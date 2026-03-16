from pydantic import BaseModel, IPvAnyAddress, validator
from uuid import UUID
from typing import Dict

class ConfidenceScores(BaseModel):
    fluent: float
    blocks: float
    interjections: float
    prolongations: float
    part_word_repetition: float
    phrase_repetition: float
    word_repetition: float
    
    @validator("*")
    def score_in_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence score must be between 0.0 and 1.0")
        return v

class PredictionCreate(BaseModel):
    audio_filename: str
    audio_duration_sec: float
    predicted_class: str
    confidence_scores: ConfidenceScores
    model_version_id: UUID
    processing_time_ms: int
    client_ip: IPvAnyAddress
    
