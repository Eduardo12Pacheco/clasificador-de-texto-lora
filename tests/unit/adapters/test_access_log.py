"""Append-only JSONL audit trail for authorised test-split reads."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stc.adapters.storage.access_log import JsonlAccessLog
from stc.domain.access import TestAccessEntry, TestAccessGrant

pytestmark = pytest.mark.unit


def make_entry(index: int, command: str = "stc evaluate --split test --final") -> TestAccessEntry:
    grant = TestAccessGrant(
        purpose="final evaluation",
        model_id=f"model-{index}",
        issued_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        model_sha256=f"sha-{index}",
    )
    return TestAccessEntry(
        grant=grant,
        command=command,
        git_sha="abc1234",
        n_examples=100 + index,
    )


def test_record_appends_one_json_line_per_entry(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "test_access_log.jsonl"
    log = JsonlAccessLog(path)

    log.record(make_entry(0))
    log.record(make_entry(1, command="stc report final"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["command"] == "stc evaluate --split test --final"
    assert first["n_examples"] == 100
    assert second["command"] == "stc report final"
    assert second["model_id"] == "model-1"
    assert second["model_sha256"] == "sha-1"
    assert first["timestamp"] == "2026-01-02T03:04:05+00:00"
    assert first["purpose"] == "final evaluation"
    assert first["git_sha"] == "abc1234"


def test_lines_are_key_sorted_and_newline_terminated(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"

    JsonlAccessLog(path).record(make_entry(0))

    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert list(json.loads(raw)) == sorted(json.loads(raw))


def test_existing_lines_are_never_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text('{"existing": true}\n', encoding="utf-8")

    JsonlAccessLog(path).record(make_entry(0))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"existing": true}'
    assert json.loads(lines[1])["n_examples"] == 100
