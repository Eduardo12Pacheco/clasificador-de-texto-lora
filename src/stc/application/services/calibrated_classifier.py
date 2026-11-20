"""Decorator that adds temperature scaling and a review policy to a classifier."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from stc.application.dto import ModelInfo
from stc.application.ports import Classifier
from stc.domain.calibration import Calibration, softmax
from stc.domain.errors import InvalidInputError
from stc.domain.labels import LabelSpace
from stc.domain.prediction import Prediction, ScoredLabel

__all__ = ["CalibratedClassifier"]


class CalibratedClassifier:
    """Wrap any :class:`Classifier` with D10 post-processing."""

    def __init__(self, inner: Classifier, calibration: Calibration) -> None:
        self._inner = inner
        self._calibration = calibration

    @property
    def calibration(self) -> Calibration:
        """The fitted calibration applied by this decorator."""
        return self._calibration

    @property
    def info(self) -> ModelInfo:
        """Describe the wrapped artifact."""
        return self._inner.info

    @property
    def labels(self) -> LabelSpace:
        """Expose the wrapped label space."""
        return self._inner.labels

    def predict_scores(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Delegate raw logits unchanged."""
        return self._inner.predict_scores(texts)

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return temperature-scaled probabilities."""
        scores = np.asarray(self._inner.predict_scores(texts), dtype=np.float64)
        return softmax(scores, T=self._calibration.temperature)

    def predict(self, texts: Sequence[str], top_k: int = 3) -> list[Prediction]:
        """Return calibrated top-k predictions with review flags."""
        if top_k < 1 or top_k > self.labels.n_intents:
            raise InvalidInputError(
                f"top_k must be between 1 and {self.labels.n_intents}, got {top_k}"
            )
        probabilities = self.predict_proba(texts)
        predictions: list[Prediction] = []
        for row in probabilities:
            ranked = np.argsort(-row, kind="stable")[:top_k]
            ranked_labels = tuple(
                ScoredLabel(
                    intent=self.labels.intents[int(index)],
                    scenario=self.labels.scenarios[self.labels.scenario_of_intent(int(index))],
                    probability=float(row[index]),
                )
                for index in ranked
            )
            best = ranked_labels[0]
            predictions.append(
                Prediction(
                    intent=best.intent,
                    scenario=best.scenario,
                    confidence=best.probability,
                    needs_review=best.probability < self._calibration.review_threshold,
                    top_k=ranked_labels,
                )
            )
        return predictions
