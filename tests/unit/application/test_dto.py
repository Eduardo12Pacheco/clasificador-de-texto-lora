"""Tests for the application DTOs."""

import pytest

from stc.application.dto import DatasetMeta
from stc.domain.errors import IntegrityError

pytestmark = pytest.mark.unit


def make_meta() -> DatasetMeta:
    return DatasetMeta(
        dataset_id="massive",
        version="1.0",
        tarball_sha256="a" * 64,
        stc_version="0.1.0",
        raw_counts={"train": 3, "dev": 1, "test": 1},
        processed_counts={"train": 2, "dev": 1, "test": 1},
        dropped_counts={"EXACT_DUPLICATE": 1},
    )


def test_dataset_meta_round_trips() -> None:
    meta = make_meta()
    assert DatasetMeta.from_dict(meta.to_dict()) == meta


def test_dataset_meta_to_dict_is_a_copy() -> None:
    meta = make_meta()
    payload = meta.to_dict()
    raw_counts = payload["raw_counts"]
    assert isinstance(raw_counts, dict)
    raw_counts["train"] = 99
    assert meta.raw_counts["train"] == 3


@pytest.mark.parametrize("field", ["dataset_id", "version", "tarball_sha256", "stc_version"])
def test_dataset_meta_rejects_non_string_scalars(field: str) -> None:
    payload = make_meta().to_dict()
    payload[field] = 7
    with pytest.raises(IntegrityError):
        DatasetMeta.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_counts", []),
        ("raw_counts", {"train": "many"}),
        ("raw_counts", {"train": True}),
        ("processed_counts", None),
        ("dropped_counts", {"EXACT_DUPLICATE": 1.5}),
    ],
)
def test_dataset_meta_rejects_malformed_counts(field: str, value: object) -> None:
    payload = make_meta().to_dict()
    payload[field] = value
    with pytest.raises(IntegrityError):
        DatasetMeta.from_dict(payload)
