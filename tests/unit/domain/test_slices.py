"""Slice masks for the natural-slice part of S9."""

from __future__ import annotations

import numpy as np
import pytest

from stc.domain.examples import Example
from stc.domain.slices import has_accent, has_digit, length_bucket, slice_masks

pytestmark = pytest.mark.unit


def test_length_buckets_use_whitespace_word_counts() -> None:
    assert length_bucket("") == "le4"
    assert length_bucket("uno dos tres cuatro") == "le4"
    assert length_bucket("uno dos tres cuatro cinco") == "5_8"
    assert length_bucket(" ".join(["x"] * 8)) == "5_8"
    assert length_bucket(" ".join(["x"] * 9)) == "9_12"
    assert length_bucket(" ".join(["x"] * 12)) == "9_12"
    assert length_bucket(" ".join(["x"] * 13)) == "ge13"


def test_digit_and_accent_flags() -> None:
    assert has_digit("pon la alarma a las 7") is True
    assert has_digit("pon la alarma") is False
    assert has_accent("música") is True
    assert has_accent("mañana") is False
    assert has_accent("despiertame") is False


def _example(
    row_id: str,
    text: str,
    *,
    scenario: str = "alarm",
    worker: str = "w1",
    intent: str | None = None,
) -> Example:
    return Example(
        id=row_id,
        text=text,
        intent=intent or f"{scenario}_set",
        scenario=scenario,
        worker_id=worker,
    )


def test_slice_masks_partition_rows_and_respect_worker_support() -> None:
    examples = [
        _example("1", "pon la alarma", scenario="alarm", worker="w1"),
        _example("2", "pon la alarma a las 7", scenario="alarm", worker="w1"),
        _example("3", "pon música de Juanes", scenario="play", worker="w2"),
        _example("4", "¿qué tiempo hace en Quito?", scenario="weather", worker="w2"),
    ]

    masks = slice_masks(examples, min_worker_support=2)

    scenario_keys = [key for key in masks if key.startswith("scenario:")]
    digit_keys = [key for key in masks if key.startswith("digit:")]
    accent_keys = [key for key in masks if key.startswith("accent:")]
    length_keys = [key for key in masks if key.startswith("length:")]

    assert scenario_keys == ["scenario:alarm", "scenario:play", "scenario:weather"]
    assert sum(masks[key].astype(int) for key in scenario_keys).tolist() == [1, 1, 1, 1]
    assert sum(masks[key].astype(int) for key in digit_keys).tolist() == [1, 1, 1, 1]
    assert sum(masks[key].astype(int) for key in accent_keys).tolist() == [1, 1, 1, 1]
    assert sum(masks[key].astype(int) for key in length_keys).tolist() == [1, 1, 1, 1]
    assert masks["digit:yes"].tolist() == [False, True, False, False]
    assert masks["accent:yes"].tolist() == [False, False, True, True]
    assert "worker:w1" in masks
    assert "worker:w2" in masks

    without_worker_support = slice_masks(examples, min_worker_support=3)
    assert not any(key.startswith("worker:") for key in without_worker_support)


def test_empty_examples_produce_no_masks() -> None:
    assert slice_masks([]) == {}


def test_mask_values_are_boolean_numpy_arrays() -> None:
    masks = slice_masks([_example("1", "hola")])
    for mask in masks.values():
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_
