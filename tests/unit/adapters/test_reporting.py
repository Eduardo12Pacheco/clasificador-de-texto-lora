"""Deterministic JSON, CSV and Markdown report writers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stc.adapters.reporting.csv_writer import CsvReportWriter
from stc.adapters.reporting.json_writer import JsonReportWriter
from stc.adapters.reporting.markdown import MarkdownReportWriter
from stc.adapters.reporting.plots import (
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_reliability_diagram,
)
from stc.domain.calibration import reliability_bins
from stc.domain.errors import MissingExtraError

pytestmark = pytest.mark.unit


def test_json_writer_sorts_keys_rounds_floats_and_ends_with_newline(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    writer = JsonReportWriter()

    writer.write_json(
        path,
        {"z": 1, "a": {"value": 0.123456, "nested": [1.23456, 2]}, "text": "música"},
    )

    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    payload = json.loads(raw)
    assert list(payload) == ["a", "text", "z"]
    assert payload["a"]["value"] == 0.1235
    assert payload["a"]["nested"] == [1.2346, 2]
    assert raw.index('"a"') < raw.index('"text"') < raw.index('"z"')


def test_json_writer_is_byte_deterministic(tmp_path: Path) -> None:
    writer = JsonReportWriter()
    payload = {"b": [0.1, 0.2], "a": {"x": 1.00005}}

    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    writer.write_json(first, payload)
    writer.write_json(second, payload)

    assert first.read_bytes() == second.read_bytes()


def test_csv_writer_rounds_floats_and_uses_unix_endings(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    writer = CsvReportWriter()

    writer.write_csv(
        path,
        ["intent", "f1", "support"],
        [["alarm_set", 0.123456, 3], ["play_music", 1.0, 0]],
    )

    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.decode().splitlines() == [
        "intent,f1,support",
        "alarm_set,0.1235,3",
        "play_music,1.0,0",
    ]


def test_csv_writer_quotes_fields_with_commas(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"

    CsvReportWriter().write_csv(path, ["text"], [["hola, mundo"]])

    assert path.read_text(encoding="utf-8").splitlines()[1] == '"hola, mundo"'


def test_markdown_writer_normalises_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    writer = MarkdownReportWriter()

    writer.write_markdown(path, "# Title\r\nBody")

    assert path.read_bytes() == b"# Title\nBody\n"


def test_markdown_writer_keeps_a_single_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "report.md"

    MarkdownReportWriter().write_markdown(path, "# Title\n\n")

    assert path.read_bytes() == b"# Title\n"


def test_plot_helpers_raise_missing_extra_without_matplotlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    bins = reliability_bins([0.5, 0.6], [True, False])
    cm = __import__("numpy").eye(2, dtype=int)

    with pytest.raises(MissingExtraError) as first:
        plot_reliability_diagram(bins, tmp_path / "reliability.png")
    with pytest.raises(MissingExtraError) as second:
        plot_confusion_matrix(cm, ("alarm_set", "play_music"), tmp_path / "confusion.png")
    with pytest.raises(MissingExtraError) as third:
        plot_per_class_f1(("alarm_set", "play_music"), [0.5, 0.6], tmp_path / "per_class.png")

    assert first.value.exit_code == 2
    assert second.value.exit_code == 2
    assert third.value.exit_code == 2
    assert "train" in first.value.message
