"""Pure factory that parses ``STC_TRACKER`` and wires injected adapters (D9)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from stc.application.ports import ExperimentTracker
from stc.domain.errors import InvalidInputError

__all__ = ["TrackerFactory", "parse_tracker_names"]

TrackerBuilder = Callable[[Path], ExperimentTracker]


def parse_tracker_names(spec: str | Sequence[str]) -> tuple[str, ...]:
    """Split ``jsonl,mlflow`` or accept a sequence, rejecting unknown names later."""
    if isinstance(spec, str):
        names = tuple(part.strip() for part in spec.split(",") if part.strip())
    else:
        names = tuple(str(part).strip() for part in spec if str(part).strip())
    return names or ("null",)


class TrackerFactory:
    """Create the tracker(s) named by configuration without importing adapters."""

    def __init__(
        self,
        builders: Mapping[str, TrackerBuilder],
        *,
        composite_builder: Callable[[Sequence[ExperimentTracker]], ExperimentTracker],
    ) -> None:
        self._builders = dict(builders)
        self._composite_builder = composite_builder

    def create(self, spec: str | Sequence[str], run_dir: Path) -> ExperimentTracker:
        """Return one tracker, a composite of many, or the Null Object fallback."""
        names = parse_tracker_names(spec)
        trackers: list[ExperimentTracker] = []
        for name in names:
            builder = self._builders.get(name)
            if builder is None:
                known = ", ".join(sorted(self._builders))
                raise InvalidInputError(f"unknown tracker: {name!r}; expected one of {known}")
            trackers.append(builder(Path(run_dir)))
        if len(trackers) == 1:
            return trackers[0]
        return self._composite_builder(trackers)
