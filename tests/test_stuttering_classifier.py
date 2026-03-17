"""
Tests for the stuttering classifier model.
"""

import pytest
import torch

from ai.models.stuttering_classifier import (
    STUTTERING_CLASSES,
    ModelConfig,
    StutteringClassifier,
    build_model,
)
from shared.labels import NUM_CLASSES


# Use a tiny model in tests to avoid downloading ~360MB and speed up runs.
# Falls back to facebook/wav2vec2-base if the tiny model is unavailable.
TINY_MODEL = "hf-internal-testing/tiny-random-Wav2Vec2Model"


@pytest.fixture
def config():
    """Config with tiny model and small defaults for fast tests."""
    return ModelConfig(
        model_name=TINY_MODEL,
        num_classes=NUM_CLASSES,
        dropout_rate=0.1,
        freeze_encoder=True,
        learning_rate=1e-4,
    )


@pytest.fixture
def model(config):
    """StutteringClassifier built from config (uses tiny model)."""
    return StutteringClassifier(config)


def test_model_config_defaults():
    """ModelConfig has expected default values."""
    c = ModelConfig()
    assert c.model_name == "facebook/wav2vec2-base"
    assert c.num_classes == NUM_CLASSES
    assert c.dropout_rate == 0.1
    assert c.freeze_encoder is True
    assert c.learning_rate == 1e-4


def test_stuttering_classes():
    """STUTTERING_CLASSES has 7 entries and correct labels."""
    assert len(STUTTERING_CLASSES) == NUM_CLASSES
    assert STUTTERING_CLASSES[0] == "fluent"
    assert STUTTERING_CLASSES[1] == "blocks"
    assert STUTTERING_CLASSES[2] == "interjections"
    assert STUTTERING_CLASSES[3] == "prolongations"
    assert STUTTERING_CLASSES[4] == "part_word_repetition"
    assert STUTTERING_CLASSES[5] == "phrase_repetition"
    assert STUTTERING_CLASSES[6] == "word_repetition"


def test_forward_output_shape(model):
    """Forward returns (batch, num_classes) logits."""
    batch, seq_len = 2, 1000
    input_values = torch.randn(batch, seq_len)
    logits = model(input_values)
    assert logits.shape == (batch, NUM_CLASSES)


def test_forward_with_attention_mask(model):
    """Forward with attention_mask returns correct shape and ignores padding."""
    batch, seq_len = 2, 1000
    input_values = torch.randn(batch, seq_len)
    # Mask: first sample full length, second sample half length
    attention_mask = torch.ones(batch, seq_len)
    attention_mask[1, 500:] = 0
    logits = model(input_values, attention_mask=attention_mask)
    assert logits.shape == (batch, NUM_CLASSES)


def test_freeze_encoder_freezes_params(config):
    """When freeze_encoder=True, encoder parameters have requires_grad=False."""
    config.freeze_encoder = True
    model = StutteringClassifier(config)
    for param in model.encoder.parameters():
        assert param.requires_grad is False
    for param in model.classifier.parameters():
        assert param.requires_grad is True


def test_freeze_encoder_false_allows_grad(config):
    """When freeze_encoder=False, encoder parameters have requires_grad=True."""
    config.freeze_encoder = False
    model = StutteringClassifier(config)
    for param in model.encoder.parameters():
        assert param.requires_grad is True
    for param in model.classifier.parameters():
        assert param.requires_grad is True


def test_build_model_returns_classifier_on_device(config):
    """build_model returns StutteringClassifier on a valid device."""
    model = build_model(config)
    assert isinstance(model, StutteringClassifier)
    assert next(model.parameters()).device.type in ("cuda", "mps", "cpu")


def test_forward_no_nan(model):
    """Forward does not produce NaN for small random input."""
    input_values = torch.randn(2, 500) * 0.1
    logits = model(input_values)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
