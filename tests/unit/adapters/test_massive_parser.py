"""Tests for the MASSIVE JSONL parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stc.adapters.data.massive_parser import parse_jsonl
from stc.domain.errors import InvalidInputError
from stc.domain.splits import Split

pytestmark = pytest.mark.unit


def row(
    row_id: str,
    partition: str = "train",
    intent: str = "alarm_set",
    scenario: str = "alarm",
    text: str = "pon una alarma",
    **extra: object,
) -> dict[str, object]:
    return {
        "id": row_id,
        "locale": "es-ES",
        "partition": partition,
        "scenario": scenario,
        "intent": intent,
        "utt": text,
        "annot_utt": f"[{intent}] {text}",
        "worker_id": "worker_1",
        "judgments": [{"worker_id": "worker_1", "intent": intent}],
        **extra,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
        encoding="utf-8",
    )
    return path


def complete_rows() -> list[dict[str, object]]:
    return [
        row("1", "train", "alarm_set", "alarm", "pon una alarma"),
        row("2", "dev", "play_music", "play", "pon música"),
        row("3", "test", "weather_query", "weather", "qué tiempo hace"),
    ]


def test_parse_groups_by_partition_and_ignores_extra_fields(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "es-ES.jsonl", complete_rows())

    splits = parse_jsonl(path)

    assert set(splits) == {Split.TRAIN, Split.DEV, Split.TEST}
    assert [example.id for example in splits[Split.TRAIN]] == ["1"]
    assert splits[Split.DEV][0].text == "pon música"
    assert splits[Split.DEV][0].scenario == "play"
    assert splits[Split.TEST][0].worker_id == "worker_1"


def test_parse_preserves_file_order(tmp_path: Path) -> None:
    rows = [row(str(index)) for index in range(5)]
    rows.append(row("dev-1", partition="dev"))
    rows.append(row("test-1", partition="test"))
    path = write_jsonl(tmp_path / "es-ES.jsonl", rows)

    assert [example.id for example in parse_jsonl(path)[Split.TRAIN]] == [
        "0",
        "1",
        "2",
        "3",
        "4",
    ]


@pytest.mark.parametrize("key", ["id", "partition", "intent", "scenario", "utt", "worker_id"])
def test_missing_required_key_is_invalid_input(tmp_path: Path, key: str) -> None:
    rows = complete_rows()
    del rows[0][key]
    path = write_jsonl(tmp_path / "es-ES.jsonl", rows)

    with pytest.raises(InvalidInputError, match="line 1"):
        parse_jsonl(path)


def test_wrong_value_type_is_invalid_input(tmp_path: Path) -> None:
    rows = complete_rows()
    rows[0]["utt"] = 42
    path = write_jsonl(tmp_path / "es-ES.jsonl", rows)

    with pytest.raises(InvalidInputError, match="utt"):
        parse_jsonl(path)


@pytest.mark.parametrize("partition", ["validation", "TRAIN", ""])
def test_unknown_partition_is_invalid_input(tmp_path: Path, partition: str) -> None:
    rows = complete_rows()
    rows[0]["partition"] = partition
    path = write_jsonl(tmp_path / "es-ES.jsonl", rows)

    with pytest.raises(InvalidInputError, match="partition"):
        parse_jsonl(path)


def test_intent_scenario_mismatch_is_invalid_input(tmp_path: Path) -> None:
    rows = complete_rows()
    rows[0]["scenario"] = "play"
    path = write_jsonl(tmp_path / "es-ES.jsonl", rows)

    with pytest.raises(InvalidInputError, match="scenario"):
        parse_jsonl(path)


def test_duplicate_id_is_invalid_input(tmp_path: Path) -> None:
    rows = complete_rows()
    rows[1]["id"] = rows[0]["id"]
    path = write_jsonl(tmp_path / "es-ES.jsonl", rows)

    with pytest.raises(InvalidInputError, match="duplicate id"):
        parse_jsonl(path)


def test_missing_partition_is_invalid_input(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "es-ES.jsonl", complete_rows()[:1])

    with pytest.raises(InvalidInputError, match="missing partition"):
        parse_jsonl(path)


def test_bad_json_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "es-ES.jsonl"
    path.write_text("{not json}\n")

    with pytest.raises(InvalidInputError, match="line 1"):
        parse_jsonl(path)


def test_blank_line_is_invalid_input(tmp_path: Path) -> None:
    path = tmp_path / "es-ES.jsonl"
    path.write_text("\n")

    with pytest.raises(InvalidInputError, match="line 1"):
        parse_jsonl(path)


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="not found"):
        parse_jsonl(tmp_path / "absent.jsonl")
