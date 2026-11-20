"""Robustness and natural-slice evaluators (S9)."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pytest

from stc.application.dto import ModelInfo
from stc.application.use_cases.robustness import (
    MACRO_F1_SUBSET_NOTE,
    RobustnessEvaluator,
    SliceEvaluator,
)
from stc.domain.calibration import Calibration
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace
from stc.domain.splits import Split

pytestmark = pytest.mark.unit

INTENTS = ("alarm_cancel", "alarm_set", "play_music", "weather_query")


class MappingClassifier:
    """Return explicit logits per text; unknown texts fall back to zeros."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping
        self.seen: list[str] = []

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            model_id="fake-model",
            kind="fake",
            version="0.0.1",
            quantization=None,
            parameters=10,
            license="MIT",
            n_intents=len(INTENTS),
            n_scenarios=3,
        )

    @property
    def labels(self) -> LabelSpace:
        return LabelSpace.from_intents(INTENTS)

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        self.seen.extend(texts)
        return np.asarray(
            [self._mapping.get(text, [0.0, 0.0, 0.0, 0.0]) for text in texts],
            dtype=np.float32,
        )


def _example(row_id: int, text: str, intent: str) -> Example:
    return Example(
        id=str(row_id),
        text=text,
        intent=intent,
        scenario=intent.split("_", maxsplit=1)[0],
        worker_id="w1",
    )


@pytest.fixture
def examples() -> list[Example]:
    return [
        _example(1, "pon una alarma", "alarm_set"),
        _example(2, "pon música", "play_music"),
        _example(3, "qué tiempo hace", "weather_query"),
        _example(4, "cancela la alarma", "alarm_cancel"),
    ]


@pytest.fixture
def classifier(examples: list[Example]) -> MappingClassifier:
    return MappingClassifier(
        {
            "pon una alarma": [0.0, 5.0, 0.0, 0.0],
            "pon música": [0.0, 0.0, 5.0, 0.0],
            "qué tiempo hace": [0.0, 0.0, 0.0, 5.0],
            "cancela la alarma": [5.0, 0.0, 0.0, 0.0],
            "PON UNA ALARMA": [0.0, 0.0, 5.0, 0.0],
            "PON MÚSICA": [0.0, 0.0, 5.0, 0.0],
            "QUÉ TIEMPO HACE": [0.0, 0.0, 5.0, 0.0],
            "CANCELA LA ALARMA": [0.0, 0.0, 5.0, 0.0],
        }
    )


def test_robustness_scores_only_the_changed_subset(
    classifier: MappingClassifier, examples: list[Example]
) -> None:
    report = RobustnessEvaluator().run(classifier, examples, ["uppercase"])

    assert report.split is Split.DEV
    assert report.n_examples == 4
    row = report.perturbations[0]
    assert row.name == "uppercase"
    assert row.n_affected == 4
    assert row.accuracy_before == pytest.approx(1.0)
    assert row.accuracy_after == pytest.approx(0.25)
    assert row.delta_accuracy == pytest.approx(-0.75)
    assert row.delta_ci_lower <= row.delta_accuracy <= row.delta_ci_upper
    assert 0.0 <= row.mean_confidence_after <= 1.0
    assert report.to_dict()["note"] == MACRO_F1_SUBSET_NOTE


def test_perturbation_with_no_changed_rows_reports_zeroes(
    classifier: MappingClassifier, examples: list[Example]
) -> None:
    report = RobustnessEvaluator().run(classifier, examples, ["regionalize:mx"])

    row = report.perturbations[0]
    assert row.n_affected == 0
    assert row.accuracy_before == 0.0
    assert row.accuracy_after == 0.0
    assert row.delta_ci_lower == 0.0


def test_calibration_changes_the_reported_confidence(
    classifier: MappingClassifier, examples: list[Example]
) -> None:
    raw = RobustnessEvaluator().run(classifier, examples, ["uppercase"])
    calibrated = RobustnessEvaluator().run(
        classifier, examples, ["uppercase"], Calibration(temperature=4.0, review_threshold=0.5)
    )

    assert calibrated.perturbations[0].mean_confidence_after < (
        raw.perturbations[0].mean_confidence_after
    )


def test_unknown_perturbation_is_rejected(
    classifier: MappingClassifier, examples: list[Example]
) -> None:
    with pytest.raises(InvalidInputError):
        RobustnessEvaluator().run(classifier, examples, ["nope"])


def test_empty_examples_are_rejected(classifier: MappingClassifier) -> None:
    with pytest.raises(InvalidInputError):
        RobustnessEvaluator().run(classifier, [], ["uppercase"])


class KeywordClassifier:
    @property
    def info(self) -> ModelInfo:
        return ModelInfo("fake", "fake", "0.0.1", None, 10, "MIT", 2, 1)

    @property
    def labels(self) -> LabelSpace:
        return LabelSpace.from_intents(("alarm_set", "play_music"))

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [[0.0, 5.0] if "music" in text else [5.0, 0.0] for text in texts],
            dtype=np.float32,
        )


def _big_examples() -> list[Example]:
    rows = []
    for index in range(220):
        intent = "alarm_set" if index % 2 == 0 else "play_music"
        keyword = "alarm" if intent == "alarm_set" else "music"
        scenario = "big" if index < 200 else "small"
        rows.append(
            Example(
                id=str(index),
                text=f"{keyword} {index}",
                intent=intent,
                scenario=scenario,
                worker_id="w1" if index % 2 == 0 else "w2",
            )
        )
    return rows


def test_slice_report_has_scenario_partitions_and_worker_spread() -> None:
    report = SliceEvaluator(min_macro_f1_support=200).run(KeywordClassifier(), _big_examples())

    by_name = {row.name: row for row in report.rows}
    assert by_name["scenario:big"].n == 200
    assert by_name["scenario:big"].accuracy == pytest.approx(1.0)
    assert by_name["scenario:big"].macro_f1 == pytest.approx(1.0)
    assert by_name["scenario:small"].macro_f1 is None
    assert report.worker_spread is not None
    assert report.worker_spread.n_workers == 2
    assert report.worker_spread.spread == pytest.approx(0.0)
    assert json.dumps(report.to_dict())  # serialisable


def test_slice_rows_have_bootstrap_intervals() -> None:
    report = SliceEvaluator().run(KeywordClassifier(), _big_examples()[:40])

    for row in report.rows:
        assert row.accuracy_ci_lower <= row.accuracy <= row.accuracy_ci_upper
