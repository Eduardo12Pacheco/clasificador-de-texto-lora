"""Tests for the packaged CC BY 4.0 fixture sample."""

from __future__ import annotations

import pytest

from stc.adapters.data.fixture_source import FixtureRepository
from stc.domain.errors import InvalidInputError
from stc.domain.splits import Split

pytestmark = pytest.mark.unit


@pytest.fixture
def repository() -> FixtureRepository:
    return FixtureRepository()


def test_fixture_counts_fit_the_d24_budget(repository: FixtureRepository) -> None:
    counts = {split: len(repository.load(split)) for split in Split}

    assert counts[Split.TRAIN] <= 360
    assert counts[Split.DEV] <= 120
    assert counts[Split.TEST] <= 120
    assert sum(counts.values()) <= 600
    assert repository.source_size_bytes() <= 120 * 1024


def test_fixture_train_covers_every_intent(repository: FixtureRepository) -> None:
    train = repository.load(Split.TRAIN)

    assert len({example.intent for example in train}) == 60
    assert all(example.scenario == example.intent.split("_", maxsplit=1)[0] for example in train)


def test_fixture_meta_matches_the_rows(repository: FixtureRepository) -> None:
    meta = repository.meta()

    assert meta.dataset_id == "massive-fixture"
    assert meta.tarball_sha256 == "fixture"
    assert meta.processed_counts == {
        split.value: len(repository.load(split)) for split in Split
    }


def test_fixture_is_read_only(repository: FixtureRepository) -> None:
    with pytest.raises(InvalidInputError, match="read-only"):
        repository.save(Split.TRAIN, [])
    with pytest.raises(InvalidInputError, match="read-only"):
        repository.save_meta(repository.meta())
