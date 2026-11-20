"""Tests for the dev-only dataset repository decorator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from stc.application.dto import DatasetMeta
from stc.application.services.dev_only_repository import DevOnlyRepository
from stc.domain.access import TestAccessGrant as AccessGrant
from stc.domain.errors import TestSplitLocked as LockedError
from stc.domain.examples import Example
from stc.domain.splits import Split

pytestmark = pytest.mark.unit


@dataclass
class FakeRepository:
    saved: list[tuple[Split, tuple[Example, ...]]] = field(default_factory=list)

    def load(self, split: Split) -> Sequence[Example]:
        return (
            Example(id="9", text=split.value, intent="alarm_set", scenario="alarm", worker_id="w"),
        )

    def save(self, split: Split, examples: Sequence[Example]) -> None:
        self.saved.append((split, tuple(examples)))

    def meta(self) -> DatasetMeta:
        return DatasetMeta("m", "1", "0" * 64, "0.1.0", {}, {}, {})

    def save_meta(self, meta: DatasetMeta) -> None:
        return None


@pytest.fixture
def grant() -> AccessGrant:
    from datetime import UTC, datetime

    return AccessGrant(
        purpose="final evaluation",
        model_id="stc-es-intent",
        issued_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_dev_and_train_are_readable() -> None:
    repository = DevOnlyRepository(FakeRepository())

    assert repository.load(Split.DEV)[0].text == "dev"
    assert repository.load(Split.TRAIN)[0].text == "train"


def test_test_split_is_locked() -> None:
    repository = DevOnlyRepository(FakeRepository())

    with pytest.raises(LockedError) as error:
        repository.load(Split.TEST)

    assert error.value.exit_code == 6


def test_load_test_needs_a_grant(grant: AccessGrant) -> None:
    repository = DevOnlyRepository(FakeRepository())

    with pytest.raises(LockedError):
        repository.load_test(None)

    assert repository.load_test(grant)[0].text == "test"


def test_writes_and_metadata_pass_through() -> None:
    inner = FakeRepository()
    repository = DevOnlyRepository(inner)

    repository.save(Split.TRAIN, [Example("1", "t", "alarm_set", "alarm", "w")])
    assert inner.saved[0][0] is Split.TRAIN
    assert repository.meta().dataset_id == "m"
