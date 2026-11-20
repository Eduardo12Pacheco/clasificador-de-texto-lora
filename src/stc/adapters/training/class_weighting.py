"""The D7 ``sqrt_inv_freq`` class-weighting rule (pure NumPy)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from stc.domain.errors import InvalidInputError

__all__ = ["sqrt_inv_freq_weights"]


def sqrt_inv_freq_weights(
    counts: Mapping[int, int] | Sequence[int],
    *,
    lower: float = 0.25,
    upper: float = 4.0,
) -> NDArray[np.float64]:
    """Return ``clip((median_count / n_c) ** 0.5, lower, upper)`` per class."""
    if isinstance(counts, Mapping):
        if not counts:
            raise InvalidInputError("class counts must not be empty")
        n_classes = max(counts) + 1
        per_class = np.zeros(n_classes, dtype=np.float64)
        for index, count in counts.items():
            if index < 0 or count < 0:
                raise InvalidInputError("class counts must be non-negative")
            per_class[index] = float(count)
    else:
        per_class = np.asarray(counts, dtype=np.float64)
        if per_class.ndim != 1 or per_class.size == 0:
            raise InvalidInputError("class counts must be a non-empty one-dimensional sequence")
        if np.any(per_class < 0):
            raise InvalidInputError("class counts must be non-negative")
    observed = per_class[per_class > 0]
    if observed.size == 0:
        raise InvalidInputError("at least one class must have a positive count")
    median = float(np.median(observed))
    weights = np.full_like(per_class, upper)
    positive = per_class > 0
    weights[positive] = np.clip(np.sqrt(median / per_class[positive]), lower, upper)
    return weights
