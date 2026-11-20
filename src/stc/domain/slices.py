"""Natural slices of S9: scenario, length, digits, accents and workers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.text import strip_accents

__all__ = ["LENGTH_BUCKETS", "has_accent", "has_digit", "length_bucket", "slice_masks"]

LENGTH_BUCKETS = ("le4", "5_8", "9_12", "ge13")


def length_bucket(text: str) -> str:
    """Bucket an utterance by its whitespace-delimited word count."""
    words = len(text.split())
    if words <= 4:
        return "le4"
    if words <= 8:
        return "5_8"
    if words <= 12:
        return "9_12"
    return "ge13"


def has_digit(text: str) -> bool:
    """Whether the utterance contains at least one decimal digit."""
    return any(character.isdigit() for character in text)


def has_accent(text: str) -> bool:
    """Whether the utterance carries a diacritic other than the tilde on ``ñ``."""
    return strip_accents(text, keep_tilde_n=True) != text


def _mask(values: Iterable[bool]) -> NDArray[np.bool_]:
    return np.fromiter(values, dtype=np.bool_)


def slice_masks(
    examples: Sequence[Example],
    min_worker_support: int = 30,
) -> dict[str, NDArray[np.bool_]]:
    """Build every S9 natural-slice mask.

    Scenario, length, digit and accent masks partition the rows; worker masks
    are only emitted for workers with at least ``min_worker_support`` rows.
    """
    if min_worker_support < 1:
        raise InvalidInputError("min_worker_support must be positive")
    if not examples:
        return {}

    masks: dict[str, NDArray[np.bool_]] = {}
    for scenario in sorted({example.scenario for example in examples}):
        masks[f"scenario:{scenario}"] = _mask(example.scenario == scenario for example in examples)
    for bucket in LENGTH_BUCKETS:
        if any(length_bucket(example.text) == bucket for example in examples):
            masks[f"length:{bucket}"] = _mask(
                length_bucket(example.text) == bucket for example in examples
            )
    masks["digit:yes"] = _mask(has_digit(example.text) for example in examples)
    masks["digit:no"] = _mask(not has_digit(example.text) for example in examples)
    masks["accent:yes"] = _mask(has_accent(example.text) for example in examples)
    masks["accent:no"] = _mask(not has_accent(example.text) for example in examples)

    counts = Counter(example.worker_id for example in examples)
    for worker in sorted(counts):
        if counts[worker] >= min_worker_support:
            masks[f"worker:{worker}"] = _mask(example.worker_id == worker for example in examples)
    return masks
