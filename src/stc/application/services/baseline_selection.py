"""Baseline model-selection policy fixed by decision D5."""

from __future__ import annotations

from collections.abc import Sequence

from stc.application.dto import BaselineCandidate
from stc.domain.errors import InvalidInputError

SELECTION_RULE = "max dev macro-F1; tie -> higher dev accuracy; tie -> fewer features"


def select_baseline_winner(candidates: Sequence[BaselineCandidate]) -> BaselineCandidate:
    """Pick the D5 winner: macro-F1, then accuracy, then fewer features.

    Python's ``max`` keeps the first maximal element, so complete ties fall back
    to the order the grid produced, keeping the choice deterministic.
    """
    if not candidates:
        raise InvalidInputError("the baseline grid produced no candidate configurations")
    return max(
        candidates,
        key=lambda candidate: (
            candidate.dev_macro_f1,
            candidate.dev_accuracy,
            -candidate.n_features,
        ),
    )
