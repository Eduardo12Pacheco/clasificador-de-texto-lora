"""Read-only access to the small CC BY 4.0 sample bundled with the package."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import resources
from importlib.resources.abc import Traversable
from typing import IO

from stc import __version__
from stc.adapters.data.jsonl_repository import example_from_payload
from stc.application.dto import DatasetMeta
from stc.domain.errors import ArtifactNotFoundError, InvalidInputError
from stc.domain.examples import Example
from stc.domain.splits import Split

_FIXTURE_DIR = ("_fixture", "massive_es_sample")


def _fixture_resource(split: Split) -> Traversable:
    return resources.files("stc").joinpath(*_FIXTURE_DIR, f"{split.value}.jsonl")


class FixtureRepository:
    """Repository over the deterministic sample shipped inside the wheel."""

    def load(self, split: Split) -> Sequence[Example]:
        """Read one fixture split from the installed package data."""
        try:
            with _fixture_resource(split).open("rb") as handle:
                return self._parse(handle, f"{split.value}.jsonl")
        except (FileNotFoundError, ModuleNotFoundError) as error:
            raise ArtifactNotFoundError(
                "bundled fixture sample is missing; run `uv run python scripts/make_fixture.py`"
            ) from error

    def save(self, split: Split, examples: Sequence[Example]) -> None:
        """The packaged fixture is immutable."""
        raise InvalidInputError("the bundled fixture sample is read-only")

    def save_meta(self, meta: DatasetMeta) -> None:
        """The packaged fixture has no writable metadata."""
        raise InvalidInputError("the bundled fixture sample is read-only")

    def meta(self) -> DatasetMeta:
        """Describe the fixture from its own rows."""
        counts = {split.value: len(self.load(split)) for split in Split}
        return DatasetMeta(
            dataset_id="massive-fixture",
            version="1.0",
            tarball_sha256="fixture",
            stc_version=__version__,
            raw_counts=counts,
            processed_counts=dict(counts),
            dropped_counts={},
        )

    def source_size_bytes(self) -> int:
        """Total size of the packaged JSONL sample in bytes."""
        total = 0
        for split in Split:
            try:
                total += len(_fixture_resource(split).read_bytes())
            except (FileNotFoundError, OSError) as error:
                raise ArtifactNotFoundError(
                    "bundled fixture sample is missing; run `uv run python scripts/make_fixture.py`"
                ) from error
        return total

    @staticmethod
    def _parse(handle: IO[bytes], name: str) -> tuple[Example, ...]:
        rows: list[Example] = []
        for line_number, raw_line in enumerate(handle.read().decode("utf-8").splitlines(), start=1):
            if not raw_line.strip():
                raise InvalidInputError(f"{name} line {line_number}: blank line")
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise InvalidInputError(f"{name} line {line_number}: invalid JSON") from error
            rows.append(example_from_payload(payload, line_number=line_number))
        return tuple(rows)
