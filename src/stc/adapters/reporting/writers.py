"""Composite report writer that fans out to the deterministic format adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from stc.adapters.reporting.csv_writer import CsvReportWriter
from stc.adapters.reporting.json_writer import JsonReportWriter
from stc.adapters.reporting.markdown import MarkdownReportWriter
from stc.domain.errors import MissingExtraError

__all__ = ["CompositeReportWriter"]


class CompositeReportWriter:
    """Implement the ``ReportWriter`` port over the format-specific adapters."""

    def __init__(
        self,
        json_writer: JsonReportWriter | None = None,
        csv_writer: CsvReportWriter | None = None,
        markdown_writer: MarkdownReportWriter | None = None,
    ) -> None:
        self._json = json_writer if json_writer is not None else JsonReportWriter()
        self._csv = csv_writer if csv_writer is not None else CsvReportWriter()
        self._markdown = markdown_writer if markdown_writer is not None else MarkdownReportWriter()

    def dumps(self, payload: Mapping[str, object]) -> str:
        """Serialise a deterministic JSON document."""
        return self._json.dumps(payload)

    def write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        """Write deterministic JSON."""
        self._json.write_json(path, payload)

    def write_csv(
        self,
        path: Path,
        header: Sequence[str],
        rows: Sequence[Sequence[object]],
    ) -> None:
        """Write a deterministic CSV table."""
        self._csv.write_csv(path, header, rows)

    def write_markdown(self, path: Path, text: str) -> None:
        """Write Markdown with normalised line endings."""
        self._markdown.write_markdown(path, text)

    def write_figure(self, path: Path, figure: object) -> None:
        """Reject direct figure writes; the final report uses its own figure port."""
        del path, figure
        raise MissingExtraError(
            "plotting needs the 'train' extra: uv tool install "
            "'spanish-text-classifier-lora[train]'"
        )
