"""Tests for the label space value object."""

import dataclasses

import numpy as np
import pytest

from stc.domain.errors import LabelSpaceError
from stc.domain.labels import LabelSpace

pytestmark = pytest.mark.unit

INTENTS = ["alarm_set", "alarm_remove", "music_play", "music_pause", "weather_query"]


def test_from_intents_sorts_and_derives_scenarios() -> None:
    labels = LabelSpace.from_intents(INTENTS)
    assert labels.intents == tuple(sorted(INTENTS))
    assert labels.scenarios == ("alarm", "music", "weather")
    assert labels.n_intents == 5
    assert labels.n_scenarios == 3


def test_intent_index_round_trips() -> None:
    labels = LabelSpace.from_intents(INTENTS)
    for index, intent in enumerate(labels.intents):
        assert labels.intent_index(intent) == index


def test_scenario_index_and_of_intent() -> None:
    labels = LabelSpace.from_intents(INTENTS)
    for index, intent in enumerate(labels.intents):
        scenario = intent.split("_", maxsplit=1)[0]
        expected = labels.scenario_index(scenario)
        assert labels.scenario_of_intent(index) == expected


def test_scenario_map_is_aligned_with_intents() -> None:
    labels = LabelSpace.from_intents(INTENTS)
    assert labels.scenario_map.dtype == np.int64
    assert labels.scenario_map.tolist() == [
        labels.scenario_index(intent.split("_", maxsplit=1)[0]) for intent in labels.intents
    ]


def test_to_scenario_indices_maps_intent_predictions() -> None:
    labels = LabelSpace.from_intents(INTENTS)
    intent_indices = np.array([0, 2, 4], dtype=np.int64)
    mapped = labels.to_scenario_indices(intent_indices)
    assert mapped.dtype == np.int64
    assert mapped.tolist() == [labels.scenario_map[index] for index in intent_indices]


def test_duplicate_intents_are_rejected() -> None:
    with pytest.raises(LabelSpaceError):
        LabelSpace.from_intents(["alarm_set", "alarm_set"])


@pytest.mark.parametrize(
    "intent",
    ["alarm", "Alarm_Set", "alarm set", "alarm__set", "_alarm_set", "alarm_set_"],
)
def test_invalid_intent_names_are_rejected(intent: str) -> None:
    with pytest.raises(LabelSpaceError):
        LabelSpace.from_intents([intent])


def test_unknown_intent_raises_label_space_error() -> None:
    labels = LabelSpace.from_intents(INTENTS)
    with pytest.raises(LabelSpaceError, match="unknown intent"):
        labels.intent_index("nope_nothing")


def test_unknown_scenario_raises_label_space_error() -> None:
    labels = LabelSpace.from_intents(INTENTS)
    with pytest.raises(LabelSpaceError, match="unknown scenario"):
        labels.scenario_index("nope")


def test_to_scenario_indices_rejects_out_of_range_indices() -> None:
    labels = LabelSpace.from_intents(INTENTS)
    with pytest.raises(LabelSpaceError):
        labels.to_scenario_indices(np.array([0, 99], dtype=np.int64))


def test_label_space_is_frozen() -> None:
    labels = LabelSpace.from_intents(INTENTS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        labels.intents = ()  # type: ignore[misc]


def test_empty_label_space_is_rejected() -> None:
    with pytest.raises(LabelSpaceError):
        LabelSpace.from_intents([])


def test_to_scenario_indices_rejects_non_integer_indices() -> None:
    labels = LabelSpace.from_intents(INTENTS)
    with pytest.raises(LabelSpaceError):
        labels.to_scenario_indices(np.array([0.0, 1.5]))


def test_to_scenario_indices_rejects_two_dimensional_indices() -> None:
    labels = LabelSpace.from_intents(INTENTS)
    with pytest.raises(LabelSpaceError):
        labels.to_scenario_indices(np.array([[0, 1]], dtype=np.int64))
