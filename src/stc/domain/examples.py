"""Immutable utterance record shared by every split."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Example:
    """One labelled utterance from the MASSIVE dataset."""

    id: str
    text: str
    intent: str
    scenario: str
    worker_id: str
