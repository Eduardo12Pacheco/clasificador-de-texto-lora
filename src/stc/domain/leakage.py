"""Leakage audit and train-only cleaning policy (decision D3)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from stc.domain.examples import Example
from stc.domain.text import normalize_for_dedup


class DropReason(StrEnum):
    """Why a training row was removed by the cleaning policy."""

    OVERLAP_WITH_EVAL = "OVERLAP_WITH_EVAL"
    LABEL_CONFLICT_DUPLICATE = "LABEL_CONFLICT_DUPLICATE"
    EXACT_DUPLICATE = "EXACT_DUPLICATE"


@dataclass(frozen=True, slots=True)
class DroppedRow:
    """A training row id and the reason it was dropped."""

    id: str
    reason: DropReason


@dataclass(frozen=True, slots=True)
class CleanedTrain:
    """Surviving training rows plus the audit trail of dropped rows."""

    rows: tuple[Example, ...]
    dropped: tuple[DroppedRow, ...]


@dataclass(frozen=True, slots=True)
class WorkerPartitionOverlap:
    """How many translator workers appear in more than one partition."""

    workers_in_multiple_partitions: int
    total_workers: int


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    """Counts produced by :func:`audit_splits`; dev and test are never modified."""

    train_rows: int
    dev_rows: int
    test_rows: int
    dev_test_overlap_texts: int
    dev_rows_in_train: int
    dev_rows_in_train_label_conflict: int
    test_rows_in_train: int
    test_rows_in_train_label_conflict: int
    train_duplicate_groups: int
    train_duplicate_excess_rows: int
    dev_duplicate_groups: int
    dev_duplicate_excess_rows: int
    test_duplicate_groups: int
    test_duplicate_excess_rows: int
    worker_partition_overlap: WorkerPartitionOverlap


def _normalized_texts(examples: Sequence[Example]) -> list[str]:
    return [normalize_for_dedup(example.text) for example in examples]


def _duplicate_stats(examples: Sequence[Example]) -> tuple[int, int]:
    counts = Counter(_normalized_texts(examples))
    groups = [count for count in counts.values() if count > 1]
    return len(groups), sum(count - 1 for count in groups)


def _train_intents_by_text(train: Sequence[Example]) -> dict[str, set[str]]:
    intents_by_text: dict[str, set[str]] = {}
    for example in train:
        intents_by_text.setdefault(normalize_for_dedup(example.text), set()).add(example.intent)
    return intents_by_text


def _rows_in_train(
    examples: Sequence[Example],
    train_intents_by_text: dict[str, set[str]],
) -> tuple[int, int]:
    rows = 0
    conflicts = 0
    for example in examples:
        train_intents = train_intents_by_text.get(normalize_for_dedup(example.text))
        if train_intents is None:
            continue
        rows += 1
        if example.intent not in train_intents:
            conflicts += 1
    return rows, conflicts


def _worker_partition_overlap(
    train: Sequence[Example],
    dev: Sequence[Example],
    test: Sequence[Example],
) -> WorkerPartitionOverlap:
    partitions = [
        {example.worker_id for example in train},
        {example.worker_id for example in dev},
        {example.worker_id for example in test},
    ]
    workers = set().union(*partitions)
    shared = sum(1 for worker in workers if sum(worker in part for part in partitions) > 1)
    return WorkerPartitionOverlap(
        workers_in_multiple_partitions=shared,
        total_workers=len(workers),
    )


def audit_splits(
    train: Sequence[Example],
    dev: Sequence[Example],
    test: Sequence[Example],
) -> LeakageAudit:
    """Compute every leakage/duplication count documented in S2."""
    train_intents_by_text = _train_intents_by_text(train)
    dev_rows_in_train, dev_conflicts = _rows_in_train(dev, train_intents_by_text)
    test_rows_in_train, test_conflicts = _rows_in_train(test, train_intents_by_text)

    train_groups, train_excess = _duplicate_stats(train)
    dev_groups, dev_excess = _duplicate_stats(dev)
    test_groups, test_excess = _duplicate_stats(test)

    dev_texts = set(_normalized_texts(dev))
    test_texts = set(_normalized_texts(test))

    return LeakageAudit(
        train_rows=len(train),
        dev_rows=len(dev),
        test_rows=len(test),
        dev_test_overlap_texts=len(dev_texts & test_texts),
        dev_rows_in_train=dev_rows_in_train,
        dev_rows_in_train_label_conflict=dev_conflicts,
        test_rows_in_train=test_rows_in_train,
        test_rows_in_train_label_conflict=test_conflicts,
        train_duplicate_groups=train_groups,
        train_duplicate_excess_rows=train_excess,
        dev_duplicate_groups=dev_groups,
        dev_duplicate_excess_rows=dev_excess,
        test_duplicate_groups=test_groups,
        test_duplicate_excess_rows=test_excess,
        worker_partition_overlap=_worker_partition_overlap(train, dev, test),
    )


def clean_train(
    train: Sequence[Example],
    dev: Sequence[Example],
    test: Sequence[Example],
) -> CleanedTrain:
    """Apply the D3 cleaning policy to the training split only."""
    ordered = sorted(train, key=lambda example: int(example.id))
    evaluation_texts = {*_normalized_texts(dev), *_normalized_texts(test)}

    dropped: dict[str, DropReason] = {}
    groups: dict[str, list[Example]] = {}
    for example in ordered:
        text = normalize_for_dedup(example.text)
        if text in evaluation_texts:
            dropped[example.id] = DropReason.OVERLAP_WITH_EVAL
        else:
            groups.setdefault(text, []).append(example)

    kept: list[Example] = []
    for group in groups.values():
        if len({example.intent for example in group}) > 1:
            for example in group:
                dropped[example.id] = DropReason.LABEL_CONFLICT_DUPLICATE
            continue
        kept.append(group[0])
        for example in group[1:]:
            dropped[example.id] = DropReason.EXACT_DUPLICATE

    dropped_rows = tuple(
        DroppedRow(id=row_id, reason=dropped[row_id]) for row_id in sorted(dropped, key=int)
    )
    return CleanedTrain(rows=tuple(kept), dropped=dropped_rows)
