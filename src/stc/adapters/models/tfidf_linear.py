"""TF-IDF + linear baseline: D5 grid, dev metrics and the classifier adapter."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from stc import __version__
from stc.application.dto import BaselineCandidate, BaselineGrid, ModelInfo
from stc.application.services.baseline_selection import select_baseline_winner
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace
from stc.domain.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    present_labels,
)

__all__ = [
    "TfidfLinearClassifier",
    "TfidfLinearTrainer",
    "load_baseline",
    "parameters_from_pipeline",
]


def _pair(value: object, field_name: str) -> tuple[int, int]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence) or len(value) != 2:
        raise InvalidInputError(f"{field_name} must be a [min, max] pair")
    lower, upper = value
    if (
        isinstance(lower, bool)
        or isinstance(upper, bool)
        or not isinstance(lower, int)
        or not isinstance(upper, int)
        or lower < 1
        or lower > upper
    ):
        raise InvalidInputError(f"{field_name} must be a positive [min, max] integer pair")
    return lower, upper


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidInputError(f"{field_name} must be a positive integer")
    return value


def _positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise InvalidInputError(f"{field_name} must be a positive number")
    return float(value)


def _class_weight(value: object) -> str | None:
    if value in (None, "none"):
        return None
    if value == "balanced":
        return "balanced"
    raise InvalidInputError(f"unsupported class_weight: {value!r}")


def _build_pipeline(params: Mapping[str, object], *, random_state: int) -> Pipeline:
    word_ngram = _pair(params.get("word_ngram"), "word_ngram")
    char_ngram = _pair(params.get("char_ngram"), "char_ngram")
    classifier = params.get("classifier")
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=word_ngram,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=char_ngram,
                    sublinear_tf=True,
                    min_df=2,
                    lowercase=True,
                ),
            ),
        ]
    )
    if classifier == "linear_svc":
        estimator: object = LinearSVC(
            C=_positive_float(params.get("C"), "C"),
            class_weight=_class_weight(params.get("class_weight")),
            random_state=random_state,
        )
    elif classifier == "logistic_regression":
        estimator = LogisticRegression(
            C=_positive_float(params.get("C"), "C"),
            class_weight=_class_weight(params.get("class_weight")),
            max_iter=_positive_int(params.get("max_iter", 2000), "max_iter"),
            random_state=random_state,
        )
    else:
        raise InvalidInputError(f"unsupported baseline classifier: {classifier!r}")
    return Pipeline([("features", features), ("classifier", estimator)])


def parameters_from_pipeline(pipeline: object) -> int:
    """Count the features the fitted vectorisers produced."""
    steps = getattr(pipeline, "named_steps", None)
    if not isinstance(steps, dict):
        return 0
    features = steps.get("features")
    total = 0
    for _, transformer in getattr(features, "transformer_list", ()):
        total += len(getattr(transformer, "vocabulary_", {}))
    return total


def _labels_and_targets(
    examples: Sequence[Example], labels: LabelSpace
) -> tuple[list[str], NDArray[np.int64]]:
    return (
        [example.text for example in examples],
        np.array([labels.intent_index(example.intent) for example in examples], dtype=np.int64),
    )


@dataclass(frozen=True, slots=True)
class TfidfLinearClassifier:
    """A fitted sklearn pipeline exposed through the ``Classifier`` port."""

    pipeline: object
    labels: LabelSpace
    info: ModelInfo

    def predict_scores(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return ``decision_function`` scores as float32 logits."""
        if not texts:
            raise InvalidInputError("prediction needs at least one text")
        decision = getattr(self.pipeline, "decision_function", None)
        if decision is None:
            raise InvalidInputError("the fitted baseline pipeline cannot score texts")
        scores = np.asarray(decision(list(texts)), dtype=np.float32)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1).astype(np.float32)
        return scores


class TfidfLinearTrainer:
    """Fit the D5 grid with a deterministic seed and report per-config dev metrics."""

    def evaluate_configs(
        self,
        train: Sequence[Example],
        dev: Sequence[Example],
        labels: LabelSpace,
        grid: BaselineGrid,
        *,
        quick: bool = False,
    ) -> Sequence[BaselineCandidate]:
        """Fit every configuration, scoring each one on dev only."""
        train_texts, y_train = _labels_and_targets(train, labels)
        dev_texts, y_dev = _labels_and_targets(dev, labels)
        present = present_labels(y_dev)
        svc_candidates: list[BaselineCandidate] = []
        for word_ngram in grid.word_ngrams:
            for char_ngram in grid.char_ngrams:
                for c_value in grid.svc_c:
                    for class_weight in grid.class_weights:
                        params: dict[str, object] = {
                            "word_ngram": list(word_ngram),
                            "char_ngram": list(char_ngram),
                            "classifier": "linear_svc",
                            "C": c_value,
                            "class_weight": None if class_weight == "none" else class_weight,
                        }
                        svc_candidates.append(
                            self._evaluate_one(
                                params,
                                train_texts,
                                y_train,
                                dev_texts,
                                y_dev,
                                present,
                                n_classes=labels.n_intents,
                                random_state=grid.random_state,
                            )
                        )
                        if quick and len(svc_candidates) >= grid.quick_configs:
                            return tuple(svc_candidates)

        best_svc = select_baseline_winner(svc_candidates)
        best_pair = {
            "word_ngram": list(_pair(best_svc.params.get("word_ngram"), "word_ngram")),
            "char_ngram": list(_pair(best_svc.params.get("char_ngram"), "char_ngram")),
        }
        candidates = list(svc_candidates)
        for c_value in grid.logreg_c:
            params = {
                **best_pair,
                "classifier": "logistic_regression",
                "C": c_value,
                "class_weight": "balanced",
                "max_iter": grid.logreg_max_iter,
            }
            candidates.append(
                self._evaluate_one(
                    params,
                    train_texts,
                    y_train,
                    dev_texts,
                    y_dev,
                    present,
                    n_classes=labels.n_intents,
                    random_state=grid.random_state,
                )
            )
        return tuple(candidates)

    def fit(
        self,
        train: Sequence[Example],
        labels: LabelSpace,
        params: Mapping[str, object],
    ) -> TfidfLinearClassifier:
        """Fit one configuration and wrap it as a classifier."""
        pipeline = _build_pipeline(
            params,
            random_state=_positive_int(params.get("random_state", 13), "random_state"),
        )
        texts, targets = _labels_and_targets(train, labels)
        pipeline.fit(texts, targets)
        return TfidfLinearClassifier(
            pipeline=pipeline,
            labels=labels,
            info=ModelInfo(
                model_id="tfidf-linear-baseline",
                kind="tfidf-linear",
                version=__version__,
                quantization=None,
                parameters=parameters_from_pipeline(pipeline),
                license="MIT",
                n_intents=labels.n_intents,
                n_scenarios=labels.n_scenarios,
            ),
        )

    @staticmethod
    def _evaluate_one(
        params: Mapping[str, object],
        train_texts: Sequence[str],
        y_train: NDArray[np.int64],
        dev_texts: Sequence[str],
        y_dev: NDArray[np.int64],
        present: NDArray[np.int64],
        *,
        n_classes: int,
        random_state: int,
    ) -> BaselineCandidate:
        pipeline = _build_pipeline(params, random_state=random_state)
        started = time.perf_counter()
        pipeline.fit(train_texts, y_train)
        fit_seconds = time.perf_counter() - started
        scores = np.asarray(pipeline.decision_function(list(dev_texts)))
        predicted = np.argmax(scores, axis=1).astype(np.int64)
        matrix = confusion_matrix(y_dev, predicted, n_classes)
        return BaselineCandidate(
            params=dict(params),
            dev_accuracy=accuracy(y_dev, predicted),
            dev_macro_f1=macro_f1(matrix, present),
            n_features=parameters_from_pipeline(pipeline),
            fit_seconds=fit_seconds,
        )


def load_baseline(path: Path) -> TfidfLinearClassifier:
    """Load a persisted baseline directory through the skops store."""
    from stc.adapters.models.skops_store import SkopsBaselineStore

    classifier = SkopsBaselineStore().load(path)
    if not isinstance(classifier, TfidfLinearClassifier):
        raise InvalidInputError(f"{path} is not a tfidf-linear baseline")
    return classifier
