"""Composite tracker that fans out and isolates failing children (D23)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from stc.application.ports import ExperimentTracker

__all__ = ["CompositeTracker"]


class CompositeTracker:
    """Forward every call to all children, never letting one failure stop the rest."""

    def __init__(self, trackers: Sequence[ExperimentTracker]) -> None:
        self._trackers = tuple(trackers)
        self.errors: list[str] = []

    @property
    def children(self) -> tuple[ExperimentTracker, ...]:
        """The wrapped trackers, for tests and diagnostics."""
        return self._trackers

    def _fan_out(self, action: Callable[[ExperimentTracker], None]) -> None:
        for index, tracker in enumerate(self._trackers):
            try:
                action(tracker)
            except Exception as error:
                self.errors.append(f"{index}:{type(error).__name__}")

    def start_run(self, run_name: str, tags: Mapping[str, str]) -> None:
        """Start every child run."""
        self._fan_out(lambda tracker: tracker.start_run(run_name, tags))

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log parameters to every child."""
        self._fan_out(lambda tracker: tracker.log_params(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log metrics to every child."""
        self._fan_out(lambda tracker: tracker.log_metrics(metrics, step))

    def log_artifact(self, path: Path) -> None:
        """Attach the artifact to every child."""
        self._fan_out(lambda tracker: tracker.log_artifact(path))

    def end_run(self) -> None:
        """End every child run."""
        self._fan_out(lambda tracker: tracker.end_run())
