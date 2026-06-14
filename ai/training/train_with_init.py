"""
Wrapper around ai/training/train.py that supports:
  - INIT_CHECKPOINT_PATH: load initial model weights from a previous checkpoint
    (e.g. English pretrain → Arabic finetune)
  - CLASS_WEIGHTS: comma-separated floats applied to CrossEntropyLoss to
    counteract class imbalance (one weight per class)

Both env vars are optional. With neither set, behaves identically to train.py.

Usage:
    INIT_CHECKPOINT_PATH=/path/to/best_model.pt \
    CLASS_WEIGHTS=0.32,1.69,3.76,2.18,0.51,4.45,2.58 \
    python ai/training/train_with_init.py --config <yaml>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.training import train as train_module  # noqa: E402
from shared.labels import NUM_CLASSES  # noqa: E402


_INIT_CKPT = os.environ.get("INIT_CHECKPOINT_PATH", "").strip()
_CLASS_WEIGHTS = os.environ.get("CLASS_WEIGHTS", "").strip()


if _INIT_CKPT:
    _original_build_model = train_module._build_model

    def _build_model_with_init(cfg: dict, device: torch.device):
        model = _original_build_model(cfg, device)
        ckpt_path = Path(_INIT_CKPT)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"INIT_CHECKPOINT_PATH not found: {ckpt_path}")
        print(f"[init-ckpt] loading weights from {ckpt_path}")
        # Load to CPU first — checkpoints carry optimizer/scheduler state, and
        # mapping all of it straight to CUDA can spike VRAM before training.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[init-ckpt] missing keys ({len(missing)}): "
                  f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
        if unexpected:
            print(f"[init-ckpt] unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
        print(f"[init-ckpt] loaded {len(sd)} tensors successfully")
        return model

    train_module._build_model = _build_model_with_init


if _CLASS_WEIGHTS:
    _weights_tensor = torch.tensor(
        [float(w) for w in _CLASS_WEIGHTS.split(",")],
        dtype=torch.float32,
    )
    # Catch a misconfigured CLASS_WEIGHTS at startup, not mid-training when
    # cross_entropy fails on a weight/class shape mismatch.
    if _weights_tensor.numel() != NUM_CLASSES:
        raise ValueError(
            f"CLASS_WEIGHTS has {_weights_tensor.numel()} values but the model "
            f"has {NUM_CLASSES} classes — counts must match"
        )
    print(f"[class-weights] applying weights: {_weights_tensor.tolist()}")

    _original_cross_entropy = F.cross_entropy

    def _weighted_cross_entropy(input, target, *args, **kwargs):
        if "weight" not in kwargs or kwargs["weight"] is None:
            kwargs["weight"] = _weights_tensor.to(input.device)
        return _original_cross_entropy(input, target, *args, **kwargs)

    F.cross_entropy = _weighted_cross_entropy
    train_module.F.cross_entropy = _weighted_cross_entropy


if __name__ == "__main__":
    train_module.main()
