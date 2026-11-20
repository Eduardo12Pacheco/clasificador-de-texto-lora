"""MLflow tracker adapter backed by a local SQLite store (D9)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import mlflow

__all__ = ["MlflowTracker"]


class MlflowTracker:
    """Log params, per-epoch metrics, tags and artifacts to MLflow."""

    def __init__(self, tracking_uri: str, *, experiment_name: str = "stc") -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._active = False

    def start_run(self, run_name: str, tags: Mapping[str, str]) -> None:
        """Open an MLflow run and attach its tags."""
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        mlflow.start_run(run_name=run_name)
        mlflow.set_tags(dict(tags))
        self._active = True

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log configuration parameters (values stringified by MLflow)."""
        mlflow.log_params({key: _scalar(value) for key, value in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log one metrics step."""
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)

    def log_artifact(self, path: Path) -> None:
        """Attach a file or directory to the active run."""
        target = Path(path)
        if target.is_dir():
            mlflow.log_artifacts(str(target))
        else:
            mlflow.log_artifact(str(target))

    def end_run(self) -> None:
        """Close the active run."""
        if self._active:
            mlflow.end_run()
            self._active = False


def _scalar(value: object) -> object:
    if value is None:
        return "null"
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)
