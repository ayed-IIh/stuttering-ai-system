"""
Wrapper around ai/training/train.py that supports loading initial weights
from a previous checkpoint (e.g. English pretrain → Arabic finetune).

Usage:
    INIT_CHECKPOINT_PATH=/path/to/best_model.pt \
    python ai/training/train_with_init.py --config <yaml>

If INIT_CHECKPOINT_PATH is unset, behaves identically to plain train.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.training import train as train_module  # noqa: E402


_INIT_CKPT = os.environ.get("INIT_CHECKPOINT_PATH", "").strip()

if _INIT_CKPT:
    _original_build_model = train_module._build_model

    def _build_model_with_init(cfg: dict, device: torch.device):
        model = _original_build_model(cfg, device)
        ckpt_path = Path(_INIT_CKPT)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"INIT_CHECKPOINT_PATH not found: {ckpt_path}")
        print(f"[init-ckpt] loading weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
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


if __name__ == "__main__":
    train_module.main()
