"""Deterministic seeding for CPU/GPU training runs (D20)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["set_all_seeds"]


def set_all_seeds(seed: int) -> None:
    """Seed every generator a training run touches and request determinism."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
