"""Tests for the leakage audit and the train-only cleaning policy."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from stc.domain.examples import Example
from stc.domain.leakage import (
    CleanedTrain,
    DropReason,
    LeakageAudit,
    audit_splits,
    clean_train,
)
from stc.domain.text import normalize_for_dedup

pytestmark = pytest.mark.unit

INTENTS = [
    "alarm_set",
    "audio_volume_down",
    "audio_volume_up",
    "close_door",
    "iot_hue_lighton",
    "open_door",
    "play_music",
    "weather_query",
]


def make_example(row_id: int, text: str, intent: str, worker_id: str = "w1") -> Example:
    return Example(
        id=str(row_id),
        text=text,
        intent=intent,
        scenario=intent.split("_", maxsplit=1)[0],
        worker_id=worker_id,
    )


@pytest.fixture
def mini_splits() -> tuple[list[Example], list[Example], list[Example]]:
    train = [
        make_example(10, "Enciende la luz", "iot_hue_lighton", "w1"),
        make_example(11, "pon música", "play_music", "w1"),
        make_example(12, "Pon Música!", "play_music", "w2"),
        make_example(13, "abre la puerta", "open_door", "w2"),
        make_example(14, "Abre la puerta", "close_door", "w1"),
        make_example(15, "qué tiempo hace", "weather_query", "w2"),
        make_example(16, "¿Qué tiempo hace?", "weather_query", "w1"),
        make_example(17, "baja el volumen", "audio_volume_up", "w2"),
        make_example(18, "texto solo en test", "alarm_set", "w1"),
        make_example(20, "sube el volumen", "audio_volume_up", "w1"),
    ]
    dev = [
        make_example(100, "enciende la luz", "iot_hue_lighton", "w1"),
        make_example(101, "baja el volumen", "audio_volume_down", "w3"),
        make_example(102, "compartido entre dev y test", "alarm_set", "w2"),
    ]
    test = [
        make_example(200, "cierra la puerta", "close_door", "w3"),
        make_example(201, "texto solo en test", "weather_query", "w1"),
        make_example(202, "compartido entre dev y test", "alarm_set", "w2"),
    ]
    return train, dev, test


def test_clean_train_drops_each_reason_in_policy_order(
    mini_splits: tuple[list[Example], list[Example], list[Example]],
) -> None:
    train, dev, test = mini_splits
    cleaned = clean_train(train, dev, test)

    assert isinstance(cleaned, CleanedTrain)
    assert [row.id for row in cleaned.rows] == ["11", "15", "20"]
    assert [(dropped.id, dropped.reason) for dropped in cleaned.dropped] == [
        ("10", DropReason.OVERLAP_WITH_EVAL),
        ("12", DropReason.EXACT_DUPLICATE),
        ("13", DropReason.LABEL_CONFLICT_DUPLICATE),
        ("14", DropReason.LABEL_CONFLICT_DUPLICATE),
        ("16", DropReason.EXACT_DUPLICATE),
        ("17", DropReason.OVERLAP_WITH_EVAL),
        ("18", DropReason.OVERLAP_WITH_EVAL),
    ]


def test_clean_train_keeps_lowest_integer_id_for_duplicates() -> None:
    train = [
        make_example(10, "pon música", "play_music"),
        make_example(2, "Pon musica", "play_music"),
        make_example(30, "pon MÚSICA", "play_music"),
    ]
    cleaned = clean_train(train, [], [])
    assert [row.id for row in cleaned.rows] == ["2"]
    assert [dropped.id for dropped in cleaned.dropped] == ["10", "30"]


def test_clean_train_does_not_mutate_its_inputs(
    mini_splits: tuple[list[Example], list[Example], list[Example]],
) -> None:
    train, dev, test = mini_splits
    train_before = tuple(train)
    dev_before = tuple(dev)
    test_before = tuple(test)

    clean_train(train, dev, test)

    assert tuple(train) == train_before
    assert tuple(dev) == dev_before
    assert tuple(test) == test_before


def test_clean_train_counts_are_conserved(
    mini_splits: tuple[list[Example], list[Example], list[Example]],
) -> None:
    train, dev, test = mini_splits
    cleaned = clean_train(train, dev, test)
    assert len(cleaned.rows) + len(cleaned.dropped) == len(train)


def test_audit_splits_reports_the_documented_counts(
    mini_splits: tuple[list[Example], list[Example], list[Example]],
) -> None:
    train, dev, test = mini_splits
    audit = audit_splits(train, dev, test)

    assert isinstance(audit, LeakageAudit)
    assert (audit.train_rows, audit.dev_rows, audit.test_rows) == (10, 3, 3)
    assert audit.dev_test_overlap_texts == 1
    assert audit.dev_rows_in_train == 2
    assert audit.dev_rows_in_train_label_conflict == 1
    assert audit.test_rows_in_train == 1
    assert audit.test_rows_in_train_label_conflict == 1
    assert audit.train_duplicate_groups == 3
    assert audit.train_duplicate_excess_rows == 3
    assert audit.dev_duplicate_groups == 0
    assert audit.dev_duplicate_excess_rows == 0
    assert audit.test_duplicate_groups == 0
    assert audit.test_duplicate_excess_rows == 0
    assert audit.worker_partition_overlap.workers_in_multiple_partitions == 3
    assert audit.worker_partition_overlap.total_workers == 3


def test_audit_does_not_mutate_its_inputs(
    mini_splits: tuple[list[Example], list[Example], list[Example]],
) -> None:
    train, dev, test = mini_splits
    train_before = tuple(train)
    dev_before = tuple(dev)
    test_before = tuple(test)

    audit_splits(train, dev, test)

    assert tuple(train) == train_before
    assert tuple(dev) == dev_before
    assert tuple(test) == test_before


@given(
    train_rows=st.lists(
        st.tuples(st.text(), st.sampled_from(INTENTS)),
        max_size=40,
    ),
    dev_rows=st.lists(
        st.tuples(st.text(), st.sampled_from(INTENTS)),
        max_size=10,
    ),
    test_rows=st.lists(
        st.tuples(st.text(), st.sampled_from(INTENTS)),
        max_size=10,
    ),
)
def test_clean_train_properties(
    train_rows: list[tuple[str, str]],
    dev_rows: list[tuple[str, str]],
    test_rows: list[tuple[str, str]],
) -> None:
    train = [make_example(index, text, intent) for index, (text, intent) in enumerate(train_rows)]
    dev = [
        make_example(10_000 + index, text, intent) for index, (text, intent) in enumerate(dev_rows)
    ]
    test = [
        make_example(20_000 + index, text, intent) for index, (text, intent) in enumerate(test_rows)
    ]

    cleaned = clean_train(train, dev, test)

    assert len(cleaned.rows) + len(cleaned.dropped) == len(train)
    assert all(isinstance(dropped.reason, DropReason) for dropped in cleaned.dropped)

    eval_texts = {normalize_for_dedup(row.text) for row in [*dev, *test]}
    remaining = [normalize_for_dedup(row.text) for row in cleaned.rows]
    assert not any(text in eval_texts for text in remaining)
    assert len(remaining) == len(set(remaining))

    by_text: dict[str, set[str]] = {}
    for row in cleaned.rows:
        by_text.setdefault(normalize_for_dedup(row.text), set()).add(row.intent)
    assert all(len(intents) == 1 for intents in by_text.values())
