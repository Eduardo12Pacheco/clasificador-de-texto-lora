"""The calibrated-classifier decorator and the evaluate-model use case."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pytest

from stc.application.dto import ModelInfo
from stc.application.services.calibrated_classifier import CalibratedClassifier
from stc.application.use_cases.evaluate_model import EvaluateModel, EvaluationResult
from stc.domain.access import TestAccessEntry
from stc.domain.calibration import Calibration
from stc.domain.errors import InvalidInputError, LabelSpaceError, TestSplitLocked
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace
from stc.domain.prediction import Prediction, ScoredLabel
from stc.domain.splits import Split

pytestmark = pytest.mark.unit

INTENTS = ("alarm_set", "alarm_cancel", "play_music", "weather_query")


class FakeClassifier:
    """Scores rows from an explicit matrix, in the order the texts arrive."""

    def __init__(self, scores: Sequence[Sequence[float]]) -> None:
        self._scores = np.asarray(scores, dtype=np.float32)
        self.seen: list[str] = []

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            model_id="fake-model",
            kind="fake",
            version="0.0.1",
            quantization=None,
            parameters=42,
            license="MIT",
            n_intents=len(INTENTS),
            n_scenarios=3,
        )

    @property
    def labels(self) -> LabelSpace:
        return LabelSpace.from_intents(INTENTS)

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        self.seen.extend(texts)
        if len(texts) != len(self._scores):
            raise AssertionError(f"expected {len(self._scores)} texts, got {len(texts)}")
        return self._scores


def make_example(row_id: int, intent: str, text: str = "texto") -> Example:
    return Example(
        id=str(row_id),
        text=text,
        intent=intent,
        scenario=intent.split("_", maxsplit=1)[0],
        worker_id="w1",
    )


class RecordingAccessLog:
    def __init__(self) -> None:
        self.entries: list[TestAccessEntry] = []

    def record(self, entry: TestAccessEntry) -> None:
        self.entries.append(entry)


@pytest.fixture
def examples() -> list[Example]:
    return [
        make_example(1, "alarm_set", "pon una alarma"),
        make_example(2, "play_music", "pon música"),
        make_example(3, "weather_query", "qué tiempo hace"),
        make_example(4, "alarm_cancel", "cancela la alarma"),
    ]


@pytest.fixture
def classifier() -> FakeClassifier:
    return FakeClassifier(
        # Columns follow the sorted label space:
        # alarm_cancel, alarm_set, play_music, weather_query.
        [
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],  # wrong: weather_query predicted as alarm_set
            [3.0, 0.0, 0.0, 0.0],
        ]
    )


def test_calibrated_classifier_predicts_probabilities_and_top_k() -> None:
    classifier = FakeClassifier([[0.0, 3.0, 0.0, 0.0]])
    calibrated = CalibratedClassifier(classifier, Calibration(1.0, 0.8))

    probabilities = calibrated.predict_proba(["pon una alarma"])
    predictions = calibrated.predict(["pon una alarma"], top_k=2)

    assert probabilities.shape == (1, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0])
    assert len(predictions) == 1
    prediction = predictions[0]
    assert isinstance(prediction, Prediction)
    assert prediction.intent == "alarm_set"
    assert prediction.scenario == "alarm"
    assert prediction.confidence == pytest.approx(float(probabilities[0, 1]))
    assert prediction.needs_review is False
    assert len(prediction.top_k) == 2
    assert all(isinstance(label, ScoredLabel) for label in prediction.top_k)
    assert prediction.top_k[0].probability >= prediction.top_k[1].probability


def test_calibrated_classifier_flags_low_confidence(classifier: FakeClassifier) -> None:
    classifier = FakeClassifier([[0.1, 0.1, 0.1, 0.1]])
    calibrated = CalibratedClassifier(classifier, Calibration(1.0, 0.8))

    prediction = calibrated.predict(["dudoso"])[0]

    assert prediction.needs_review is True


def test_calibrated_classifier_delegates_info_and_labels(classifier: FakeClassifier) -> None:
    calibrated = CalibratedClassifier(classifier, Calibration(1.0, 0.5))

    assert calibrated.info.model_id == "fake-model"
    assert calibrated.labels.intents == tuple(sorted(INTENTS))


def test_calibrated_classifier_rejects_impossible_top_k(classifier: FakeClassifier) -> None:
    calibrated = CalibratedClassifier(classifier, Calibration(1.0, 0.5))

    with pytest.raises(InvalidInputError):
        calibrated.predict(["texto"], top_k=0)
    with pytest.raises(InvalidInputError):
        calibrated.predict(["texto"], top_k=99)


def test_dev_evaluation_computes_metrics_without_a_grant(
    classifier: FakeClassifier, examples: list[Example]
) -> None:
    result = EvaluateModel().run(classifier, examples, split=Split.DEV)

    assert isinstance(result, EvaluationResult)
    assert result.split is Split.DEV
    assert result.n_examples == 4
    assert result.metrics.accuracy == pytest.approx(0.75)
    assert result.metrics.scenario_accuracy == pytest.approx(0.75)
    assert result.metrics.top3_accuracy == pytest.approx(0.75)
    assert result.per_class[0].intent == "alarm_cancel"
    support = {row.intent: row.support for row in result.per_class}
    assert support["weather_query"] == 1
    assert support["alarm_cancel"] == 1
    assert result.errors[0].example_id == "3"
    assert result.errors[0].true_intent == "weather_query"
    assert result.errors[0].predicted_intent == "alarm_set"


def test_result_is_json_serialisable(classifier: FakeClassifier, examples: list[Example]) -> None:
    payload = EvaluateModel().run(classifier, examples, split=Split.DEV).to_dict()

    text = json.dumps(payload, sort_keys=True)
    assert '"macro_f1"' in text
    assert payload["split"] == "dev"
    assert payload["n_examples"] == 4
    assert len(payload["per_class"]) == 4
    assert len(payload["confusion_matrix"]) == 4


def test_calibrated_evaluation_adds_calibration_fields(
    classifier: FakeClassifier, examples: list[Example]
) -> None:
    result = EvaluateModel().run(
        classifier,
        examples,
        split=Split.DEV,
        calibration=Calibration(temperature=2.0, review_threshold=0.6),
    )

    assert result.metrics.temperature == 2.0
    assert result.metrics.review_threshold == 0.6
    assert result.metrics.calibrated_ece is not None
    assert result.metrics.uncalibrated_ece is not None
    assert result.metrics.coverage_at_threshold is not None
    assert result.to_dict()["calibrated"] is not None


def test_test_split_without_grant_raises_and_logs_nothing(
    classifier: FakeClassifier, examples: list[Example]
) -> None:
    access_log = RecordingAccessLog()

    with pytest.raises(TestSplitLocked):
        EvaluateModel().run(
            classifier,
            examples,
            split=Split.TEST,
            access_log=access_log,
        )

    assert access_log.entries == []
    assert classifier.seen == []


def test_test_split_with_grant_requires_an_access_log(
    classifier: FakeClassifier, examples: list[Example]
) -> None:
    from datetime import UTC, datetime

    from stc.domain.access import TestAccessGrant

    grant = TestAccessGrant("final evaluation", "fake-model", datetime(2026, 1, 1, tzinfo=UTC))

    with pytest.raises(InvalidInputError):
        EvaluateModel().run(classifier, examples, split=Split.TEST, grant=grant)


def test_test_split_with_grant_writes_exactly_one_log_entry(
    classifier: FakeClassifier, examples: list[Example]
) -> None:
    from datetime import UTC, datetime

    from stc.domain.access import TestAccessGrant

    grant = TestAccessGrant("final evaluation", "fake-model", datetime(2026, 1, 1, tzinfo=UTC))
    access_log = RecordingAccessLog()

    result = EvaluateModel().run(
        classifier,
        examples,
        split=Split.TEST,
        grant=grant,
        access_log=access_log,
        command="stc evaluate --split test --final",
        git_sha="abc1234",
    )

    assert result.split is Split.TEST
    assert len(access_log.entries) == 1
    entry = access_log.entries[0]
    assert entry.command == "stc evaluate --split test --final"
    assert entry.git_sha == "abc1234"
    assert entry.n_examples == 4
    assert entry.grant is grant


def test_unknown_intent_is_a_label_space_error(
    classifier: FakeClassifier, examples: list[Example]
) -> None:
    bad = [make_example(1, "not_an_intent")]

    with pytest.raises(LabelSpaceError):
        EvaluateModel().run(classifier, bad, split=Split.DEV)


def test_empty_examples_are_rejected(classifier: FakeClassifier) -> None:
    with pytest.raises(InvalidInputError):
        EvaluateModel().run(classifier, [], split=Split.DEV)


class WrongShapeClassifier(FakeClassifier):
    def __init__(self) -> None:
        super().__init__([[1.0, 0.0]])

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


def test_scores_with_the_wrong_label_count_are_rejected() -> None:
    with pytest.raises(InvalidInputError):
        EvaluateModel().run(
            WrongShapeClassifier(),
            [make_example(1, "alarm_set"), make_example(2, "play_music")],
            split=Split.DEV,
        )
