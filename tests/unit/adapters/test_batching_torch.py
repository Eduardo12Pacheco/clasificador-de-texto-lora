"""Fixed-generator batching covers every example deterministically."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stc.adapters.training.batching import shuffled_batches
from stc.domain.examples import Example

pytestmark = pytest.mark.torch


def _examples(count: int) -> list[Example]:
    return [
        Example(
            id=str(index),
            text=f"text {index}",
            intent="alarm_set",
            scenario="alarm",
            worker_id="1",
        )
        for index in range(count)
    ]


def test_batches_cover_every_example_once() -> None:
    batches = shuffled_batches(_examples(10), 3, seed=13, epoch=1)

    ids = [row.id for batch in batches for row in batch]
    assert sorted(ids) == [str(index) for index in range(10)]
    assert [len(batch) for batch in batches] == [3, 3, 3, 1]


def test_same_seed_is_reproducible_and_epoch_changes_order() -> None:
    examples = _examples(20)

    first = shuffled_batches(examples, 5, seed=13, epoch=1)
    again = shuffled_batches(examples, 5, seed=13, epoch=1)
    later = shuffled_batches(examples, 5, seed=13, epoch=2)

    assert first == again
    assert first != later


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        shuffled_batches(_examples(3), 0, seed=1, epoch=0)
