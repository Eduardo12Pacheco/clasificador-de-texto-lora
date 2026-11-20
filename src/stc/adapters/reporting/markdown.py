"""Deterministic Markdown reports."""

from __future__ import annotations

from pathlib import Path

__all__ = ["MarkdownReportWriter"]


class MarkdownReportWriter:
    """Write Markdown with normalised line endings and one trailing newline."""

    def write_markdown(self, path: Path, text: str) -> None:
        """Write ``text`` to ``path``, normalising line endings."""
        normalised = text.replace("\r\n", "\n").replace("\r", "\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalised.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
