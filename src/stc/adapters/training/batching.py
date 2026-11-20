"""Deterministic shuffled mini-batches shared by LoRA and full fine-tuning."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from stc.domain.examples import Example

__all__ = ["shuffled_batches"]


def shuffled_batches(
    examples: Sequence[Example],
    batch_size: int,
    *,
    seed: int,
    epoch: int,
) -> list[list[Example]]:
    """Shuffle with a fixed generator so a seed reproduces the same batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    order = torch.randperm(len(examples), generator=generator).tolist()
    return [
        [examples[index] for index in order[start : start + batch_size]]
        for start in range(0, len(order), batch_size)
    ]
