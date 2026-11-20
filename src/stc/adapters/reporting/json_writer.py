"""Deterministic JSON reports: sorted keys, rounded floats and UTF-8."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

__all__ = ["JsonReportWriter", "round_floats"]


def round_floats(value: object) -> object:
    """Recursively round floats to four decimals, leaving other values intact."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("reports cannot contain NaN or infinite floats")
        return round(value, 4)
    if isinstance(value, np.generic):
        return round_floats(value.item())
    if isinstance(value, Mapping):
        return {str(key): round_floats(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [round_floats(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


class JsonReportWriter:
    """Write compact, key-sorted JSON with a trailing newline."""

    def dumps(self, payload: Mapping[str, object]) -> str:
        """Serialise a report deterministically."""
        rounded = round_floats(dict(payload))
        return json.dumps(rounded, sort_keys=True, indent=2, ensure_ascii=False) + "\n"

    def write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Write ``payload`` to ``path``, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.dumps(payload), encoding="utf-8", newline="\n")
