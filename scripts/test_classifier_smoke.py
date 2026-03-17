#!/usr/bin/env python3
"""
Quick smoke test for the stuttering classifier (no pytest required).

Run from repo root: python scripts/test_classifier_smoke.py
"""

import sys
from pathlib import Path

# Add project root so "ai.models" resolves
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from ai.models.stuttering_classifier import (
    STUTTERING_CLASSES,
    ModelConfig,
    build_model,
)
from shared.labels import NUM_CLASSES


def main():
    print("Using device:", "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")

    # Tiny model for fast smoke test (no large download)
    config = ModelConfig(
        model_name="hf-internal-testing/tiny-random-Wav2Vec2Model",
        num_classes=NUM_CLASSES,
        dropout_rate=0.1,
        freeze_encoder=True,
    )
    print("Building model...")
    model = build_model(config)
    print("Model on:", next(model.parameters()).device)

    batch, seq_len = 2, 1000
    x = torch.randn(batch, seq_len) * 0.1
    print(f"Forward: input {x.shape} -> ", end="")
    logits = model(x)
    print(f"output {logits.shape}")
    assert logits.shape == (batch, NUM_CLASSES), logits.shape
    assert not torch.isnan(logits).any()

    print("Classes:", STUTTERING_CLASSES)
    print("Smoke test OK.")


if __name__ == "__main__":
    main()
