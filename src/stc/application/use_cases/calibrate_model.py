"""Fit temperature scaling and the review threshold on dev (D10)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from stc.application.ports import Classifier
from stc.domain.calibration import (
    Calibration,
    choose_review_threshold,
    ece,
    fit_temperature,
    nll,
    selective_metrics,
    softmax,
)
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example

__all__ = ["CalibrateModel", "CalibrationReport"]


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Fitted calibration plus the dev evidence stored next to a bundle."""

    calibration: Calibration
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return the JSON payload written to ``calibration.json``."""
        return dict(self.payload)


class CalibrateModel:
    """Fit one temperature and one review threshold from dev scores."""

    def __init__(self, *, target_selective_accuracy: float = 0.95) -> None:
        self._target = target_selective_accuracy

    def run(self, classifier: Classifier, examples: Sequence[Example]) -> CalibrationReport:
        """Fit and report calibration for one classifier on dev."""
        if not examples:
            raise InvalidInputError("calibration needs at least one dev example")
        labels = classifier.labels
        texts = [example.text for example in examples]
        y_true = np.asarray(
            [labels.intent_index(example.intent) for example in examples], dtype=np.int64
        )
        scores = np.asarray(classifier.predict_scores(texts), dtype=np.float64)
        if scores.shape != (len(examples), labels.n_intents):
            raise InvalidInputError("classifier returned scores with an unexpected shape")

        temperature = fit_temperature(scores, y_true)
        raw = softmax(scores)
        calibrated = softmax(scores, T=temperature)
        predicted = np.argmax(calibrated, axis=1)
        correct = predicted == y_true
        raw_confidence = raw.max(axis=1)
        confidence = calibrated.max(axis=1)
        review_threshold = choose_review_threshold(confidence, correct, target=self._target)
        selective = selective_metrics(confidence, correct, review_threshold)

        calibration = Calibration(
            temperature=float(temperature),
            review_threshold=float(review_threshold),
        )
        payload: dict[str, object] = {
            "schema_version": 1,
            "temperature": calibration.temperature,
            "review_threshold": calibration.review_threshold,
            "target_selective_accuracy": self._target,
            "dev_coverage_at_threshold": selective.coverage,
            "selective_accuracy": selective.selective_accuracy,
            "ece_dev_before": ece(raw_confidence, correct),
            "ece_dev_after": ece(confidence, correct),
            "nll_dev_before": nll(raw, y_true),
            "nll_dev_after": nll(calibrated, y_true),
            "n_dev": len(examples),
        }
        return CalibrationReport(calibration=calibration, payload=payload)
