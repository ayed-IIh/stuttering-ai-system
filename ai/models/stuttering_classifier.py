"""Stuttering classifier using a pretrained speech encoder and a classification head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class ModelConfig:
    """Configuration for selecting encoder and output classes."""

    encoder_name: str = "facebook/wav2vec2-base"
    num_classes: int = 5
    dropout: float = 0.2


class StutteringClassifier(nn.Module):
    """Audio -> encoder -> temporal pooling -> classifier logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Swap encoder_name to a HuBERT checkpoint to run HuBERT experiments.
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.num_classes),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return class logits for a batch of audio waveforms.

        Args:
            input_values: Tensor `[batch, time]`.
            attention_mask: Optional tensor `[batch, time]`.
        """
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)

        # Mean pooling across time steps creates a fixed-size embedding per sample.
        embeddings = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(embeddings)
        return logits


if __name__ == "__main__":
    # Minimal sanity check for module initialization.
    model = StutteringClassifier(ModelConfig())
    print(f"Loaded classifier with encoder: {model.config.encoder_name}")