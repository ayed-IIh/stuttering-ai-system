"""Starter training script for stuttering detection models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 16


def train(config: TrainConfig) -> None:
    """Run a minimal training loop placeholder."""
    print(f"Starting training: epochs={config.epochs}, lr={config.learning_rate}, batch={config.batch_size}")
    # TODO: add dataset loading, model definition, optimizer, and evaluation.
    print("Training pipeline placeholder completed.")


if __name__ == "__main__":
    train(TrainConfig())
