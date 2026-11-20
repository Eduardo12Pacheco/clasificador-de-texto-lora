"""Deterministic CSV reports with Unix line endings."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from pathlib import Path

from stc.adapters.reporting.json_writer import round_floats

__all__ = ["CsvReportWriter"]


class CsvReportWriter:
    """Write CSV tables with rounded floats and ``\\n`` endings."""

    def write_csv(
        self,
        path: Path,
        header: Sequence[str],
        rows: Sequence[Sequence[object]],
    ) -> None:
        """Write ``rows`` under ``header`` to ``path``."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([round_floats(value) for value in row])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")
