"""Parser for the official MASSIVE ``es-ES.jsonl`` partition file."""

from __future__ import annotations

import json
from pathlib import Path

from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.splits import Split


def parse_jsonl(path: Path) -> dict[Split, tuple[Example, ...]]:
    """Parse the single MASSIVE JSONL file into one tuple per partition.

    The raw file mixes all partitions and carries annotation fields that the
    classifier does not need; only the fields of :class:`Example` are kept.
    """
    source = Path(path)
    if not source.is_file():
        raise InvalidInputError(f"dataset file not found: {source.name}")

    grouped: dict[Split, list[Example]] = {split: [] for split in Split}
    seen_ids: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                raise InvalidInputError(f"line {line_number}: blank line in {source.name}")
            payload = _load_object(raw, line_number)
            row_id = _required_string(payload, "id", line_number)
            partition = _required_string(payload, "partition", line_number)
            intent = _required_string(payload, "intent", line_number)
            scenario = _required_string(payload, "scenario", line_number)
            text = _required_string(payload, "utt", line_number)
            worker_id = _required_string(payload, "worker_id", line_number)

            try:
                split = Split(partition)
            except ValueError as error:
                raise InvalidInputError(
                    f"line {line_number}: unknown partition {partition!r}"
                ) from error

            if intent.split("_", maxsplit=1)[0] != scenario:
                raise InvalidInputError(
                    f"line {line_number}: scenario {scenario!r} does not match intent {intent!r}"
                )

            if row_id in seen_ids:
                raise InvalidInputError(f"line {line_number}: duplicate id {row_id!r}")
            seen_ids.add(row_id)
            grouped[split].append(
                Example(
                    id=row_id,
                    text=text,
                    intent=intent,
                    scenario=scenario,
                    worker_id=worker_id,
                )
            )

    missing = [split.value for split in Split if not grouped[split]]
    if missing:
        raise InvalidInputError(f"missing partition(s) in {source.name}: {', '.join(missing)}")
    return {split: tuple(rows) for split, rows in grouped.items()}


def _required_string(payload: dict[str, object], field_name: str, line_number: int) -> str:
    if field_name not in payload:
        raise InvalidInputError(f"line {line_number}: missing required field {field_name!r}")
    value = payload[field_name]
    if not isinstance(value, str):
        raise InvalidInputError(f"line {line_number}: field {field_name!r} must be a string")
    return value


def _load_object(raw: str, line_number: int) -> dict[str, object]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise InvalidInputError(f"line {line_number}: invalid JSON") from error
    if not isinstance(payload, dict):
        raise InvalidInputError(f"line {line_number}: expected a JSON object")
    return payload


class MassiveParser:
    """Adapter object implementing :class:`stc.application.ports.DatasetParser`."""

    def parse(self, path: Path) -> dict[Split, tuple[Example, ...]]:
        """Parse the pinned MASSIVE JSONL file."""
        return parse_jsonl(path)
