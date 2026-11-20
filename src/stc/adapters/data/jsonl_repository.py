"""Deterministic JSONL persistence for processed dataset splits."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from stc.application.dto import DatasetMeta
from stc.domain.errors import ArtifactNotFoundError, IntegrityError
from stc.domain.examples import Example
from stc.domain.splits import Split

_ROW_KEYS = ("id", "text", "intent", "scenario", "worker_id")


def example_from_payload(payload: object, *, line_number: int) -> Example:
    """Build an :class:`Example` from a processed-row object, validating keys."""
    if not isinstance(payload, dict):
        raise IntegrityError(f"line {line_number}: expected a JSON object")
    values: dict[str, str] = {}
    for key in _ROW_KEYS:
        value = payload.get(key)
        if not isinstance(value, str):
            raise IntegrityError(f"line {line_number}: field {key!r} must be a string")
        values[key] = value
    return Example(
        id=values["id"],
        text=values["text"],
        intent=values["intent"],
        scenario=values["scenario"],
        worker_id=values["worker_id"],
    )


class JsonlDatasetRepository:
    """Store each split as newline-delimited JSON plus one ``meta.json``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, split: Split) -> Path:
        return self.root / f"{split.value}.jsonl"

    def save(self, split: Split, examples: Sequence[Example]) -> None:
        """Write one split atomically, preserving the given order."""
        path = self._path(split)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for example in examples:
                    row = {
                        "id": example.id,
                        "text": example.text,
                        "intent": example.intent,
                        "scenario": example.scenario,
                        "worker_id": example.worker_id,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def load(self, split: Split) -> Sequence[Example]:
        """Read one split, rejecting malformed rows."""
        path = self._path(split)
        if not path.is_file():
            raise ArtifactNotFoundError(
                f"{split.value} split not found at {path}; run `stc data prepare`"
            )
        rows: list[Example] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    raise IntegrityError(f"{path.name} line {line_number}: blank line")
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise IntegrityError(f"{path.name} line {line_number}: invalid JSON") from error
                rows.append(example_from_payload(payload, line_number=line_number))
        return tuple(rows)

    def meta(self) -> DatasetMeta:
        """Read the provenance metadata written by the prepare pipeline."""
        path = self.root / "meta.json"
        if not path.is_file():
            raise ArtifactNotFoundError(
                f"processed dataset metadata not found at {path}; run `stc data prepare`"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IntegrityError("meta.json is not valid JSON") from error
        if not isinstance(payload, dict):
            raise IntegrityError("meta.json must be a JSON object")
        return DatasetMeta.from_dict(payload)

    def save_meta(self, meta: DatasetMeta) -> None:
        """Write provenance metadata deterministically."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "meta.json"
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(meta.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
