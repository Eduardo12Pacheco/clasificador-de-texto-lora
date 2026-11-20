"""Tests for the processed JSONL dataset repository."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stc.adapters.data.jsonl_repository import JsonlDatasetRepository
from stc.application.dto import DatasetMeta
from stc.domain.errors import ArtifactNotFoundError, IntegrityError
from stc.domain.examples import Example
from stc.domain.splits import Split

pytestmark = pytest.mark.unit


def example(row_id: str = "1", text: str = "pon una alarma") -> Example:
    return Example(
        id=row_id,
        text=text,
        intent="alarm_set",
        scenario="alarm",
        worker_id="worker_1",
    )


def meta() -> DatasetMeta:
    return DatasetMeta(
        dataset_id="massive",
        version="1.0",
        tarball_sha256="a" * 64,
        stc_version="0.1.0",
        raw_counts={"train": 1, "dev": 0, "test": 0},
        processed_counts={"train": 1, "dev": 0, "test": 0},
        dropped_counts={},
    )


def test_split_round_trip_is_byte_deterministic(tmp_path: Path) -> None:
    repository = JsonlDatasetRepository(tmp_path)
    rows = [example("1"), example("2", "pon música")]

    repository.save(Split.TRAIN, rows)
    first = (tmp_path / "train.jsonl").read_bytes()
    repository.save(Split.TRAIN, rows)

    assert (tmp_path / "train.jsonl").read_bytes() == first
    assert repository.load(Split.TRAIN) == tuple(rows)
    assert first.count(b"\n") == 2
    assert "música".encode() in first


def test_saved_row_schema(tmp_path: Path) -> None:
    repository = JsonlDatasetRepository(tmp_path)
    repository.save(Split.DEV, [example()])

    payload = json.loads((tmp_path / "dev.jsonl").read_text(encoding="utf-8"))

    assert list(payload) == ["id", "text", "intent", "scenario", "worker_id"]
    assert payload["text"] == "pon una alarma"


def test_missing_split_file_raises_artifact_not_found(tmp_path: Path) -> None:
    repository = JsonlDatasetRepository(tmp_path)

    with pytest.raises(ArtifactNotFoundError, match="stc data prepare"):
        repository.load(Split.TRAIN)


def test_malformed_split_file_raises_integrity_error(tmp_path: Path) -> None:
    repository = JsonlDatasetRepository(tmp_path)
    (tmp_path / "train.jsonl").write_text('{"id": "1"}\n')

    with pytest.raises(IntegrityError, match="line 1"):
        repository.load(Split.TRAIN)


def test_meta_round_trip(tmp_path: Path) -> None:
    repository = JsonlDatasetRepository(tmp_path)
    repository.save_meta(meta())

    assert repository.meta() == meta()


def test_missing_meta_raises_artifact_not_found(tmp_path: Path) -> None:
    repository = JsonlDatasetRepository(tmp_path)

    with pytest.raises(ArtifactNotFoundError, match="stc data prepare"):
        repository.meta()


def test_corrupt_meta_raises_integrity_error(tmp_path: Path) -> None:
    repository = JsonlDatasetRepository(tmp_path)
    (tmp_path / "meta.json").write_text("{not json}")

    with pytest.raises(IntegrityError, match=r"meta\.json"):
        repository.meta()


def test_meta_with_wrong_shape_raises_integrity_error(tmp_path: Path) -> None:
    repository = JsonlDatasetRepository(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps({"dataset_id": "massive"}))

    with pytest.raises(IntegrityError):
        repository.meta()
