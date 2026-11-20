"""Tests for the example value object and the split enum."""

import dataclasses

import pytest

from stc.domain.examples import Example
from stc.domain.splits import Split

pytestmark = pytest.mark.unit


def test_example_is_frozen_and_slotted() -> None:
    example = Example(
        id="1",
        text="enciende la luz",
        intent="iot_hue_lighton",
        scenario="iot",
        worker_id="w1",
    )
    assert example.text == "enciende la luz"
    assert not hasattr(example, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        example.text = "otra"  # type: ignore[misc]


def test_split_values_are_stable() -> None:
    assert Split.TRAIN == "train"
    assert Split.DEV == "dev"
    assert Split.TEST == "test"
    assert {split.value for split in Split} == {"train", "dev", "test"}
