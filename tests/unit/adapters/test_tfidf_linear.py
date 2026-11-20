"""TF-IDF + linear baseline: grid expansion, fitting and prediction."""

from __future__ import annotations

import numpy as np
import pytest

from stc.adapters.models.tfidf_linear import (
    TfidfLinearClassifier,
    TfidfLinearTrainer,
    parameters_from_pipeline,
)
from stc.application.dto import BaselineGrid
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace

pytestmark = pytest.mark.unit

INTENTS = (
    "alarm_cancel",
    "alarm_set",
    "music_dislike",
    "music_play",
    "weather_forecast",
    "weather_query",
)

PROTOTYPES = {
    "alarm_cancel": "cancela la alarma de las siete",
    "alarm_set": "pon una alarma para mañana temprano",
    "music_dislike": "no me gusta esta canción",
    "music_play": "pon música de Juanes",
    "weather_forecast": "qué tiempo hará mañana en Quito",
    "weather_query": "cómo está el clima ahora",
}


def make_examples() -> tuple[list[Example], list[Example], LabelSpace]:
    train: list[Example] = []
    dev: list[Example] = []
    identifier = 0
    for intent, text in PROTOTYPES.items():
        for index in range(3):
            identifier += 1
            train.append(
                Example(
                    id=str(identifier),
                    text=f"{text} variant {index}",
                    intent=intent,
                    scenario=intent.split("_", maxsplit=1)[0],
                    worker_id="w1",
                )
            )
        identifier += 1
        dev.append(
            Example(
                id=str(identifier),
                text=text,
                intent=intent,
                scenario=intent.split("_", maxsplit=1)[0],
                worker_id="w1",
            )
        )
    return train, dev, LabelSpace.from_intents(INTENTS)


def test_full_grid_expands_to_forty_two_configs() -> None:
    train, dev, labels = make_examples()
    trainer = TfidfLinearTrainer()

    candidates = trainer.evaluate_configs(train, dev, labels, BaselineGrid.default())

    assert len(candidates) == 42
    classifiers = [candidate.params["classifier"] for candidate in candidates]
    assert classifiers.count("linear_svc") == 40
    assert classifiers.count("logistic_regression") == 2
    assert all(0.0 <= candidate.dev_macro_f1 <= 1.0 for candidate in candidates)
    assert all(candidate.n_features > 0 for candidate in candidates)
    assert all(candidate.fit_seconds >= 0.0 for candidate in candidates)


def test_quick_grid_only_fits_the_first_two_svc_configs() -> None:
    train, dev, labels = make_examples()
    trainer = TfidfLinearTrainer()

    candidates = trainer.evaluate_configs(train, dev, labels, BaselineGrid.default(), quick=True)

    assert len(candidates) == 2
    assert {candidate.params["classifier"] for candidate in candidates} == {"linear_svc"}


def test_fit_returns_a_classifier_with_calibrated_shape() -> None:
    train, dev, labels = make_examples()
    trainer = TfidfLinearTrainer()
    params = {
        "word_ngram": [1, 2],
        "char_ngram": [2, 5],
        "classifier": "linear_svc",
        "C": 0.5,
        "class_weight": None,
    }

    classifier = trainer.fit(train, labels, params)

    assert isinstance(classifier, TfidfLinearClassifier)
    assert classifier.labels.intents == labels.intents
    assert classifier.info.kind == "tfidf-linear"
    scores = classifier.predict_scores([example.text for example in dev])
    assert scores.shape == (len(dev), labels.n_intents)
    assert scores.dtype == np.float32


def test_two_fits_with_the_same_seed_are_identical() -> None:
    train, dev, labels = make_examples()
    trainer = TfidfLinearTrainer()
    params = {
        "word_ngram": [1, 2],
        "char_ngram": [2, 5],
        "classifier": "linear_svc",
        "C": 0.5,
        "class_weight": "balanced",
    }

    first = trainer.fit(train, labels, params)
    second = trainer.fit(train, labels, params)

    np.testing.assert_array_equal(
        first.predict_scores([example.text for example in dev]),
        second.predict_scores([example.text for example in dev]),
    )


def test_logistic_regression_config_is_supported() -> None:
    train, dev, labels = make_examples()
    trainer = TfidfLinearTrainer()
    params = {
        "word_ngram": [1, 2],
        "char_ngram": [2, 5],
        "classifier": "logistic_regression",
        "C": 10.0,
        "class_weight": "balanced",
        "max_iter": 2000,
    }

    classifier = trainer.fit(train, labels, params)

    assert classifier.predict_scores([example.text for example in dev]).shape == (
        len(dev),
        labels.n_intents,
    )


def test_unknown_classifier_is_rejected() -> None:
    train, _, labels = make_examples()
    trainer = TfidfLinearTrainer()

    with pytest.raises(InvalidInputError):
        trainer.fit(
            train,
            labels,
            {
                "word_ngram": [1, 2],
                "char_ngram": [2, 5],
                "classifier": "naive_bayes",
            },
        )


def test_parameters_from_pipeline_counts_vocabulary_features() -> None:
    train, _, labels = make_examples()
    classifier = TfidfLinearTrainer().fit(
        train,
        labels,
        {
            "word_ngram": [1, 2],
            "char_ngram": [2, 5],
            "classifier": "linear_svc",
            "C": 0.5,
            "class_weight": None,
        },
    )

    assert parameters_from_pipeline(classifier.pipeline) == classifier.info.parameters
    assert parameters_from_pipeline(object()) == 0


def test_classifier_rejects_empty_text_batches() -> None:
    train, _, labels = make_examples()
    classifier = TfidfLinearTrainer().fit(
        train,
        labels,
        {
            "word_ngram": [1, 2],
            "char_ngram": [2, 5],
            "classifier": "linear_svc",
            "C": 0.5,
            "class_weight": None,
        },
    )

    with pytest.raises(InvalidInputError):
        classifier.predict_scores([])
