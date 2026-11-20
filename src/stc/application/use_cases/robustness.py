"""Robustness and natural-slice evaluation (S9)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stc.application.ports import Classifier
from stc.domain.calibration import Calibration, softmax
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.metrics import accuracy, confusion_matrix, macro_f1, present_labels
from stc.domain.perturbations import PERTURBATIONS, apply_perturbation
from stc.domain.slices import slice_masks
from stc.domain.splits import Split
from stc.domain.stats import bootstrap_ci, paired_bootstrap_delta

__all__ = [
    "MACRO_F1_SUBSET_NOTE",
    "PerturbationResult",
    "RobustnessEvaluator",
    "RobustnessReport",
    "SliceEvaluator",
    "SliceReport",
    "SliceRow",
    "WorkerSpread",
]

MACRO_F1_SUBSET_NOTE = (
    "macro-F1 is not reported on perturbation subsets: the affected subsets are too small "
    "for per-class metrics, so only accuracy and calibrated confidence are compared."
)


def _label_indices(classifier: Classifier, examples: Sequence[Example]) -> NDArray[np.int64]:
    return np.array(
        [classifier.labels.intent_index(example.intent) for example in examples], dtype=np.int64
    )


def _scores(classifier: Classifier, texts: Sequence[str]) -> NDArray[np.float64]:
    labels = classifier.labels
    scores = np.asarray(classifier.predict_scores(texts), dtype=np.float64)
    if scores.shape != (len(texts), labels.n_intents):
        raise InvalidInputError(
            f"classifier returned scores with shape {scores.shape}, "
            f"expected {(len(texts), labels.n_intents)}"
        )
    if not np.all(np.isfinite(scores)):
        raise InvalidInputError("classifier returned non-finite scores")
    return scores


def _probabilities(
    classifier: Classifier,
    texts: Sequence[str],
    calibration: Calibration | None,
) -> NDArray[np.float64]:
    temperature = calibration.temperature if calibration is not None else 1.0
    return softmax(_scores(classifier, texts), T=temperature)


@dataclass(frozen=True, slots=True)
class PerturbationResult:
    """Before/after evidence for one perturbation on its affected subset."""

    name: str
    n_affected: int
    accuracy_before: float
    accuracy_after: float
    delta_accuracy: float
    delta_ci_lower: float
    delta_ci_upper: float
    mean_confidence_before: float
    mean_confidence_after: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable row."""
        return {
            "accuracy_after": self.accuracy_after,
            "accuracy_before": self.accuracy_before,
            "delta_accuracy": self.delta_accuracy,
            "delta_ci_lower": self.delta_ci_lower,
            "delta_ci_upper": self.delta_ci_upper,
            "mean_confidence_after": self.mean_confidence_after,
            "mean_confidence_before": self.mean_confidence_before,
            "n_affected": self.n_affected,
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    """The S9 perturbation table for one classifier."""

    split: Split
    n_examples: int
    perturbations: tuple[PerturbationResult, ...]
    note: str = MACRO_F1_SUBSET_NOTE

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON payload."""
        return {
            "schema_version": 1,
            "split": self.split.value,
            "n_examples": self.n_examples,
            "note": self.note,
            "perturbations": [row.to_dict() for row in self.perturbations],
        }


class RobustnessEvaluator:
    """Apply each D11 perturbation and score the subset it actually changed."""

    def run(
        self,
        classifier: Classifier,
        examples: Sequence[Example],
        names: Sequence[str],
        calibration: Calibration | None = None,
        *,
        split: Split = Split.DEV,
    ) -> RobustnessReport:
        """Evaluate every named perturbation; unknown names raise an error."""
        if not examples:
            raise InvalidInputError("robustness evaluation needs at least one example")
        unknown = [name for name in names if name not in PERTURBATIONS]
        if unknown:
            raise InvalidInputError(
                f"unknown perturbation(s): {', '.join(unknown)}; "
                f"expected {', '.join(PERTURBATIONS)}"
            )
        texts = [example.text for example in examples]
        y_true = _label_indices(classifier, examples)
        base_probabilities = _probabilities(classifier, texts, calibration)
        base_predicted = np.argmax(base_probabilities, axis=1).astype(np.int64)
        base_confidence = base_probabilities.max(axis=1)

        rows: list[PerturbationResult] = []
        for name in names:
            perturbed = [apply_perturbation(name, text) for text in texts]
            changed = np.array(
                [after != before for before, after in zip(texts, perturbed, strict=True)],
                dtype=np.bool_,
            )
            n_affected = int(np.sum(changed))
            if n_affected == 0:
                rows.append(
                    PerturbationResult(
                        name=name,
                        n_affected=0,
                        accuracy_before=0.0,
                        accuracy_after=0.0,
                        delta_accuracy=0.0,
                        delta_ci_lower=0.0,
                        delta_ci_upper=0.0,
                        mean_confidence_before=0.0,
                        mean_confidence_after=0.0,
                    )
                )
                continue

            subset_texts = [text for text, keep in zip(perturbed, changed, strict=True) if keep]
            after_probabilities = _probabilities(classifier, subset_texts, calibration)
            after_predicted = np.argmax(after_probabilities, axis=1).astype(np.int64)
            before_predicted = base_predicted[changed]
            subset_true = y_true[changed]
            delta = paired_bootstrap_delta(accuracy, subset_true, after_predicted, before_predicted)
            rows.append(
                PerturbationResult(
                    name=name,
                    n_affected=n_affected,
                    accuracy_before=accuracy(subset_true, before_predicted),
                    accuracy_after=accuracy(subset_true, after_predicted),
                    delta_accuracy=delta.estimate,
                    delta_ci_lower=delta.lower,
                    delta_ci_upper=delta.upper,
                    mean_confidence_before=float(np.mean(base_confidence[changed])),
                    mean_confidence_after=float(np.mean(after_probabilities.max(axis=1))),
                )
            )
        return RobustnessReport(split=split, n_examples=len(examples), perturbations=tuple(rows))


@dataclass(frozen=True, slots=True)
class SliceRow:
    """Accuracy and, above the support floor, macro-F1 for one natural slice."""

    name: str
    n: int
    accuracy: float
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    macro_f1: float | None
    macro_f1_ci_lower: float | None
    macro_f1_ci_upper: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable row."""
        return {
            "accuracy": self.accuracy,
            "accuracy_ci_lower": self.accuracy_ci_lower,
            "accuracy_ci_upper": self.accuracy_ci_upper,
            "macro_f1": self.macro_f1,
            "macro_f1_ci_lower": self.macro_f1_ci_lower,
            "macro_f1_ci_upper": self.macro_f1_ci_upper,
            "n": self.n,
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class WorkerSpread:
    """The spread of accuracy across workers with enough rows."""

    n_workers: int
    minimum: float
    maximum: float
    spread: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "maximum": self.maximum,
            "minimum": self.minimum,
            "n_workers": self.n_workers,
            "spread": self.spread,
        }


@dataclass(frozen=True, slots=True)
class SliceReport:
    """The S9 natural-slice table for one classifier."""

    split: Split
    n_examples: int
    rows: tuple[SliceRow, ...]
    worker_spread: WorkerSpread | None
    min_macro_f1_support: int = 200

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON payload."""
        return {
            "schema_version": 1,
            "split": self.split.value,
            "n_examples": self.n_examples,
            "min_macro_f1_support": self.min_macro_f1_support,
            "worker_spread": (
                self.worker_spread.to_dict() if self.worker_spread is not None else None
            ),
            "slices": [row.to_dict() for row in self.rows],
        }


def _macro_f1_metric(
    n_classes: int, labels: NDArray[np.int64]
) -> Callable[[NDArray[np.int64], NDArray[np.int64]], float]:
    def metric(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
        return macro_f1(confusion_matrix(y_true, y_pred, n_classes), labels)

    return metric


class SliceEvaluator:
    """Score every natural slice with bootstrap confidence intervals."""

    def __init__(self, *, min_macro_f1_support: int = 200) -> None:
        if min_macro_f1_support < 1:
            raise InvalidInputError("min_macro_f1_support must be positive")
        self._min_macro_f1_support = min_macro_f1_support

    def run(
        self,
        classifier: Classifier,
        examples: Sequence[Example],
        calibration: Calibration | None = None,
        *,
        split: Split = Split.DEV,
        min_worker_support: int = 30,
    ) -> SliceReport:
        """Evaluate each scenario/length/digit/accent/worker slice."""
        if not examples:
            raise InvalidInputError("slice evaluation needs at least one example")
        texts = [example.text for example in examples]
        y_true = _label_indices(classifier, examples)
        probabilities = _probabilities(classifier, texts, calibration)
        y_pred = np.argmax(probabilities, axis=1).astype(np.int64)
        present = present_labels(y_true)
        macro_metric = _macro_f1_metric(classifier.labels.n_intents, present)

        rows: list[SliceRow] = []
        for name, mask in slice_masks(examples, min_worker_support=min_worker_support).items():
            n = int(np.sum(mask))
            if n == 0:
                continue
            true_slice = y_true[mask]
            predicted_slice = y_pred[mask]
            accuracy_ci = bootstrap_ci(accuracy, true_slice, predicted_slice)
            macro: float | None = None
            macro_ci_lower: float | None = None
            macro_ci_upper: float | None = None
            if n >= self._min_macro_f1_support:
                macro_ci = bootstrap_ci(macro_metric, true_slice, predicted_slice)
                macro = macro_ci.estimate
                macro_ci_lower = macro_ci.lower
                macro_ci_upper = macro_ci.upper
            rows.append(
                SliceRow(
                    name=name,
                    n=n,
                    accuracy=accuracy_ci.estimate,
                    accuracy_ci_lower=accuracy_ci.lower,
                    accuracy_ci_upper=accuracy_ci.upper,
                    macro_f1=macro,
                    macro_f1_ci_lower=macro_ci_lower,
                    macro_f1_ci_upper=macro_ci_upper,
                )
            )
        rows.sort(key=lambda row: row.name)
        worker_accuracies = [row.accuracy for row in rows if row.name.startswith("worker:")]
        worker_spread = None
        if len(worker_accuracies) >= 2:
            minimum = min(worker_accuracies)
            maximum = max(worker_accuracies)
            worker_spread = WorkerSpread(
                n_workers=len(worker_accuracies),
                minimum=minimum,
                maximum=maximum,
                spread=maximum - minimum,
            )
        return SliceReport(
            split=split,
            n_examples=len(examples),
            rows=tuple(rows),
            worker_spread=worker_spread,
            min_macro_f1_support=self._min_macro_f1_support,
        )
