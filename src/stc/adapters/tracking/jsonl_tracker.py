"""Default dependency-free tracker: one JSONL event stream per run (D9)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

__all__ = ["JsonlTracker"]


class JsonlTracker:
    """Append params, metrics and artifact paths as deterministic JSON lines."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")

    def _append(self, event: Mapping[str, object]) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def start_run(self, run_name: str, tags: Mapping[str, str]) -> None:
        """Record the run header."""
        self._append({"event": "start", "run_name": run_name, "tags": dict(tags)})

    def log_params(self, params: Mapping[str, object]) -> None:
        """Record configuration parameters."""
        self._append({"event": "params", "params": dict(params)})

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record one metrics step."""
        payload: dict[str, object] = {
            "event": "metrics",
            "metrics": {key: float(value) for key, value in metrics.items()},
        }
        if step is not None:
            payload["step"] = step
        self._append(payload)

    def log_artifact(self, path: Path) -> None:
        """Record that an artifact belongs to this run."""
        self._append({"event": "artifact", "path": str(Path(path))})

    def end_run(self) -> None:
        """Record the end of the run."""
        self._append({"event": "end"})
