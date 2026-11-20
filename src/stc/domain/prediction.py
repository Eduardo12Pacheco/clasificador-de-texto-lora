"""Prediction value objects shared by the CLI, API and UI adapters."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Prediction", "ScoredLabel"]


@dataclass(frozen=True, slots=True)
class ScoredLabel:
    """One candidate intent with its probability and derived scenario."""

    intent: str
    scenario: str
    probability: float


@dataclass(frozen=True, slots=True)
class Prediction:
    """The calibrated decision for one utterance."""

    intent: str
    scenario: str
    confidence: float
    needs_review: bool
    top_k: tuple[ScoredLabel, ...]
