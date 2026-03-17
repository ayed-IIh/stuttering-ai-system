"""
Stuttering speech classifier built on Wav2Vec2.

Output classes: fluent(0), blocks(1), interjections(2), prolongations(3),
part_word_repetition(4), phrase_repetition(5), word_repetition(6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from shared.labels import CLASS_LABELS, NUM_CLASSES

# Backward-compatible alias retained for existing imports/tests.
STUTTERING_CLASSES = list(CLASS_LABELS)


@dataclass
class ModelConfig:
    """Configuration for building and training the stuttering classifier."""

    model_name: str = "facebook/wav2vec2-base"
    num_classes: int = NUM_CLASSES
    dropout_rate: float = 0.1
    freeze_encoder: bool = True
    learning_rate: float = 1e-4


def _get_device() -> torch.device:
    """Resolve compute device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: ModelConfig) -> StutteringClassifier:
    """
    Build a StutteringClassifier from config and place it on the appropriate device.

    Device order: cuda (if available) > mps (if available) > cpu.
    """
    device = _get_device()
    model = StutteringClassifier(config)
    model = model.to(device)
    return model


class StutteringClassifier(nn.Module):
    """
    Wav2Vec2-based classifier for 7-way stuttering/disfluency detection.

    Uses mean pooling over encoder hidden states, then dropout and a linear
    projection to class logits.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Wav2Vec2Model.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size

        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(hidden_size, config.num_classes)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_values: Raw waveform or extracted features per frame.
                Shape: (batch, sequence_length).
            attention_mask: Optional mask for valid frames (1 = valid, 0 = pad).
                Shape: (batch, sequence_length). If None, all positions are used.

        Returns:
            Class logits. Shape: (batch, num_classes).
            Classes: fluent(0), blocks(1), interjections(2), prolongations(3),
            part_word_repetition(4), phrase_repetition(5), word_repetition(6).
        """
        outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (batch, enc_seq_len, hidden_size)
        enc_seq_len = hidden_states.size(1)

        # Mean pooling over time; use attention_mask to ignore padding when provided
        if attention_mask is not None:
            # Encoder downsamples time; align mask to encoder sequence length
            if attention_mask.size(1) != enc_seq_len:
                mask_2d = attention_mask.unsqueeze(1).float()  # (batch, 1, input_len)
                mask_2d = nn.functional.interpolate(
                    mask_2d, size=enc_seq_len, mode="linear", align_corners=False
                )
                attention_mask = (mask_2d.squeeze(1) > 0.5).float()
            mask = attention_mask.unsqueeze(-1).float()  # (batch, enc_seq_len, 1)
            sum_hidden = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
