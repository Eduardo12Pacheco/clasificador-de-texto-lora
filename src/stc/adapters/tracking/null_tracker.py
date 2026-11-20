"""Null Object tracker used when a command runs without tracking."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

__all__ = ["NullTracker"]


class NullTracker:
    """Accept every tracking call and do nothing."""

    def start_run(self, run_name: str, tags: Mapping[str, str]) -> None:
        """Ignore the run header."""

    def log_params(self, params: Mapping[str, object]) -> None:
        """Ignore parameters."""

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Ignore metrics."""

    def log_artifact(self, path: Path) -> None:
        """Ignore artifacts."""

    def end_run(self) -> None:
        """Ignore the run end."""
