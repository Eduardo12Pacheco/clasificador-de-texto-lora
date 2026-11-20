"""Evaluation use case: every metric of S6-S8 in one audited read."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stc.application.ports import AccessLog, Classifier
from stc.domain.access import TestAccessEntry, TestAccessGrant, require_test_grant
from stc.domain.calibration import (
    Calibration,
    brier,
    ece,
    nll,
    selective_metrics,
    softmax,
)
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_prf,
    present_labels,
    scenario_accuracy,
    topk_accuracy,
    weighted_f1,
)
from stc.domain.splits import Split

__all__ = ["EvaluateModel", "EvaluationMetrics", "EvaluationResult", "PerClassRow"]


@dataclass(frozen=True, slots=True)
class PerClassRow:
    """One row of the per-class precision/recall/F1 table (S7)."""

    intent: str
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable row."""
        return {
            "f1": self.f1,
            "intent": self.intent,
            "precision": self.precision,
            "recall": self.recall,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Intent-level metrics plus calibration evidence for one model."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    top3_accuracy: float
    scenario_accuracy: float
    uncalibrated_ece: float
    uncalibrated_nll: float
    uncalibrated_brier: float
    calibrated_ece: float | None
    calibrated_nll: float | None
    calibrated_brier: float | None
    temperature: float | None
    review_threshold: float | None
    coverage_at_threshold: float | None
    selective_accuracy: float | None

    @property
    def ece(self) -> float:
        """Calibrated ECE when available, otherwise the raw one."""
        return self.calibrated_ece if self.calibrated_ece is not None else self.uncalibrated_ece

    @property
    def calibrated(self) -> bool:
        """Whether temperature scaling and a review threshold were applied."""
        return self.temperature is not None


@dataclass(frozen=True, slots=True)
class EvaluationError:
    """One misclassified example, used by the error analysis (S10)."""

    example_id: str
    text: str
    true_intent: str
    predicted_intent: str
    confidence: float
    uncalibrated_confidence: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable error record."""
        return {
            "confidence": self.confidence,
            "example_id": self.example_id,
            "predicted_intent": self.predicted_intent,
            "text": self.text,
            "true_intent": self.true_intent,
            "uncalibrated_confidence": self.uncalibrated_confidence,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Everything the report tools need from one evaluation run."""

    split: Split
    n_examples: int
    metrics: EvaluationMetrics
    intents: tuple[str, ...]
    per_class: tuple[PerClassRow, ...]
    confusion: NDArray[np.int64]
    errors: tuple[EvaluationError, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON payload shared by CLI and report tools."""
        payload: dict[str, object] = {
            "schema_version": 1,
            "split": self.split.value,
            "n_examples": self.n_examples,
            "accuracy": self.metrics.accuracy,
            "macro_f1": self.metrics.macro_f1,
            "weighted_f1": self.metrics.weighted_f1,
            "top3_accuracy": self.metrics.top3_accuracy,
            "scenario_accuracy": self.metrics.scenario_accuracy,
            "ece": self.metrics.ece,
            "uncalibrated_ece": self.metrics.uncalibrated_ece,
            "uncalibrated_nll": self.metrics.uncalibrated_nll,
            "uncalibrated_brier": self.metrics.uncalibrated_brier,
            "temperature": self.metrics.temperature,
            "review_threshold": self.metrics.review_threshold,
            "coverage_at_threshold": self.metrics.coverage_at_threshold,
            "selective_accuracy": self.metrics.selective_accuracy,
            "calibrated": (
                {
                    "ece": self.metrics.calibrated_ece,
                    "nll": self.metrics.calibrated_nll,
                    "brier": self.metrics.calibrated_brier,
                }
                if self.metrics.calibrated
                else None
            ),
            "uncalibrated": {
                "ece": self.metrics.uncalibrated_ece,
                "nll": self.metrics.uncalibrated_nll,
                "brier": self.metrics.uncalibrated_brier,
            },
            "intents": list(self.intents),
            "per_class": [row.to_dict() for row in self.per_class],
            "confusion_matrix": self.confusion.tolist(),
            "errors": [error.to_dict() for error in self.errors],
        }
        return payload


def _scores_matrix(
    classifier: Classifier, texts: Sequence[str], n_intents: int
) -> NDArray[np.float64]:
    scores = np.asarray(classifier.predict_scores(texts), dtype=np.float64)
    if scores.shape != (len(texts), n_intents):
        raise InvalidInputError(
            f"classifier returned scores with shape {scores.shape}, "
            f"expected {(len(texts), n_intents)}"
        )
    if not np.all(np.isfinite(scores)):
        raise InvalidInputError("classifier returned non-finite scores")
    return scores


class EvaluateModel:
    """Score a classifier on a split and compute every S6-S8 metric."""

    def run(
        self,
        classifier: Classifier,
        examples: Sequence[Example],
        *,
        split: Split = Split.DEV,
        calibration: Calibration | None = None,
        grant: TestAccessGrant | None = None,
        access_log: AccessLog | None = None,
        command: str = "",
        git_sha: str = "unknown",
    ) -> EvaluationResult:
        """Evaluate ``examples``; the test split needs a grant and an access log."""
        require_test_grant(split, grant)
        if not examples:
            raise InvalidInputError("evaluation needs at least one example")
        if split is Split.TEST and access_log is None:
            raise InvalidInputError("a final evaluation must write an access-log entry")

        labels = classifier.labels
        texts = [example.text for example in examples]
        y_true = np.array(
            [labels.intent_index(example.intent) for example in examples], dtype=np.int64
        )
        scores = _scores_matrix(classifier, texts, labels.n_intents)

        raw_probabilities = softmax(scores)
        temperature = calibration.temperature if calibration is not None else 1.0
        probabilities = softmax(scores, T=temperature)

        predicted = np.argmax(probabilities, axis=1).astype(np.int64)
        matrix = confusion_matrix(y_true, predicted, labels.n_intents)
        report = per_class_prf(matrix)
        present = present_labels(y_true)
        confidence = probabilities.max(axis=1)
        correct = predicted == y_true
        raw_confidence = raw_probabilities.max(axis=1)

        selective = (
            selective_metrics(confidence, correct, calibration.review_threshold)
            if calibration is not None
            else None
        )
        metrics = EvaluationMetrics(
            accuracy=accuracy(y_true, predicted),
            macro_f1=macro_f1(matrix, present),
            weighted_f1=weighted_f1(matrix, present),
            top3_accuracy=topk_accuracy(probabilities, y_true, 3),
            scenario_accuracy=scenario_accuracy(y_true, predicted, labels.scenario_map),
            uncalibrated_ece=ece(raw_confidence, correct),
            uncalibrated_nll=nll(raw_probabilities, y_true),
            uncalibrated_brier=brier(raw_probabilities, y_true),
            calibrated_ece=ece(confidence, correct) if calibration is not None else None,
            calibrated_nll=nll(probabilities, y_true) if calibration is not None else None,
            calibrated_brier=brier(probabilities, y_true) if calibration is not None else None,
            temperature=calibration.temperature if calibration is not None else None,
            review_threshold=calibration.review_threshold if calibration is not None else None,
            coverage_at_threshold=selective.coverage if selective is not None else None,
            selective_accuracy=selective.selective_accuracy if selective is not None else None,
        )

        errors = tuple(
            EvaluationError(
                example_id=example.id,
                text=example.text,
                true_intent=labels.intents[int(y_true[index])],
                predicted_intent=labels.intents[int(predicted[index])],
                confidence=float(confidence[index]),
                uncalibrated_confidence=float(raw_confidence[index]),
            )
            for index, example in enumerate(examples)
            if not correct[index]
        )
        per_class = tuple(
            PerClassRow(
                intent=intent,
                precision=float(report.precision[index]),
                recall=float(report.recall[index]),
                f1=float(report.f1[index]),
                support=int(report.support[index]),
            )
            for index, intent in enumerate(labels.intents)
        )

        if split is Split.TEST:
            if grant is None or access_log is None:
                raise InvalidInputError("a final evaluation must write an access-log entry")
            access_log.record(
                TestAccessEntry(
                    grant=grant,
                    command=command,
                    git_sha=git_sha,
                    n_examples=len(examples),
                )
            )

        return EvaluationResult(
            split=split,
            n_examples=len(examples),
            metrics=metrics,
            intents=labels.intents,
            per_class=per_class,
            confusion=matrix,
            errors=errors,
        )


def evaluation_summary(payload: Mapping[str, object]) -> str:
    """Render a one-line human summary used by the CLI."""
    return (
        f"accuracy={payload['accuracy']} macro_f1={payload['macro_f1']} "
        f"top3={payload['top3_accuracy']} ece={payload['ece']}"
    )
