"""Value objects for scored predictions."""

from __future__ import annotations

import pytest

from stc.domain.prediction import Prediction, ScoredLabel

pytestmark = pytest.mark.unit


def test_prediction_carries_ranked_labels() -> None:
    top_k = (
        ScoredLabel(intent="alarm_set", scenario="alarm", probability=0.7),
        ScoredLabel(intent="alarm_cancel", scenario="alarm", probability=0.2),
    )

    prediction = Prediction(
        intent="alarm_set",
        scenario="alarm",
        confidence=0.7,
        needs_review=False,
        top_k=top_k,
    )

    assert prediction.intent == "alarm_set"
    assert prediction.top_k[0].probability == 0.7
    assert prediction.top_k[1].intent == "alarm_cancel"


def test_scored_label_and_prediction_are_frozen() -> None:
    label = ScoredLabel(intent="a_b", scenario="a", probability=0.5)
    prediction = Prediction("a_b", "a", 0.5, True, (label,))

    with pytest.raises(AttributeError):
        label.probability = 0.1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        prediction.intent = "c_d"  # type: ignore[misc]
