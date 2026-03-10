"""Phase 1 training entrypoint for stuttering audio classification."""

from __future__ import annotations

from dataclasses import dataclass

from ai.models.stuttering_classifier import ModelConfig, StutteringClassifier


@dataclass
class TrainConfig:
    """Starter training config; expand with paths and optimizer settings."""

    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 8
    encoder_name: str = "facebook/wav2vec2-base"
    num_classes: int = 5


def train(config: TrainConfig) -> None:
    """Skeleton training flow for the team to implement incrementally."""
    model = StutteringClassifier(
        ModelConfig(
            encoder_name=config.encoder_name,
            num_classes=config.num_classes,
        )
    )

    print("Starting Phase 1 training...")
    print(f"Encoder: {config.encoder_name}")
    print(f"Epochs: {config.epochs}, Batch size: {config.batch_size}, Learning rate: {config.learning_rate}")

    # TODO(Ali): connect dataset manifests and DataLoader objects.
    # TODO(Adan): add optimizer, scheduler, loss function, and training/validation loops.
    # TODO(Adan): log metrics and save best model checkpoints.
    _ = model
    print("Training template ready for implementation.")


if __name__ == "__main__":
    train(TrainConfig())