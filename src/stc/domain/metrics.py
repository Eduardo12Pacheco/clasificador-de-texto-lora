"""Confusion-based classification metrics (decisions D2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stc.domain.errors import InvalidInputError

__all__ = [
    "ConfusionEntry",
    "PerClassMetrics",
    "accuracy",
    "confusion_matrix",
    "macro_f1",
    "per_class_prf",
    "present_labels",
    "scenario_accuracy",
    "top_confusions",
    "topk_accuracy",
    "weighted_f1",
]


def _as_label_array(values: object, name: str, *, allow_empty: bool = False) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise InvalidInputError(f"{name} must be a one-dimensional array")
    if array.size == 0 and not allow_empty:
        raise InvalidInputError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise InvalidInputError(f"{name} must contain integer labels")
    return np.asarray(array, dtype=np.int64)


def _as_confusion(cm: object) -> NDArray[np.int64]:
    matrix = np.asarray(cm)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise InvalidInputError("confusion matrix must be square")
    if matrix.size == 0:
        raise InvalidInputError("confusion matrix must not be empty")
    if not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0):
        raise InvalidInputError("confusion matrix must contain non-negative integer counts")
    return np.asarray(matrix, dtype=np.int64)


def _as_scores(scores: object) -> NDArray[np.float64]:
    matrix = np.asarray(scores)
    if matrix.ndim != 2:
        raise InvalidInputError("scores must be a two-dimensional array")
    if not np.issubdtype(matrix.dtype, np.number):
        raise InvalidInputError("scores must be numeric")
    if not np.all(np.isfinite(matrix)):
        raise InvalidInputError("scores must be finite")
    return np.asarray(matrix, dtype=np.float64)


def confusion_matrix(
    y_true: object,
    y_pred: object,
    n_classes: int,
) -> NDArray[np.int64]:
    """Count ``(true, predicted)`` pairs, including classes never observed."""
    if n_classes <= 0:
        raise InvalidInputError("n_classes must be positive")
    true = _as_label_array(y_true, "y_true")
    predicted = _as_label_array(y_pred, "y_pred")
    if true.shape != predicted.shape:
        raise InvalidInputError("y_true and y_pred must have the same length")
    if np.any(true < 0) or np.any(true >= n_classes):
        raise InvalidInputError("y_true contains a label outside the label space")
    if np.any(predicted < 0) or np.any(predicted >= n_classes):
        raise InvalidInputError("y_pred contains a label outside the label space")
    flat = np.bincount(true * n_classes + predicted, minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes).astype(np.int64, copy=False)


def accuracy(y_true: object, y_pred: object) -> float:
    """Fraction of exactly matching labels."""
    true = _as_label_array(y_true, "y_true")
    predicted = _as_label_array(y_pred, "y_pred")
    if true.shape != predicted.shape:
        raise InvalidInputError("y_true and y_pred must have the same length")
    return float(np.mean(true == predicted))


def present_labels(y_true: object) -> NDArray[np.int64]:
    """Sorted label indices with at least one true example (D2 macro-F1 set)."""
    true = _as_label_array(y_true, "y_true", allow_empty=True)
    return np.unique(true).astype(np.int64, copy=False)


@dataclass(frozen=True, slots=True)
class PerClassMetrics:
    """Precision, recall, F1 and support for every class index."""

    precision: NDArray[np.float64]
    recall: NDArray[np.float64]
    f1: NDArray[np.float64]
    support: NDArray[np.int64]


def per_class_prf(cm: object) -> PerClassMetrics:
    """Per-class precision, recall, F1 and support with ``zero_division=0``."""
    matrix = _as_confusion(cm)
    diagonal = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.int64)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(diagonal),
        where=denominator > 0,
    )
    return PerClassMetrics(precision=precision, recall=recall, f1=f1, support=support)


def _validated_labels(cm: NDArray[np.int64], labels: object) -> NDArray[np.int64]:
    indices = _as_label_array(labels, "labels")
    if np.any(indices < 0) or np.any(indices >= cm.shape[0]):
        raise InvalidInputError("labels contains an index outside the confusion matrix")
    return indices


def macro_f1(cm: object, labels: object) -> float:
    """Unweighted mean F1 over the supplied label indices (D2)."""
    matrix = _as_confusion(cm)
    indices = _validated_labels(matrix, labels)
    report = per_class_prf(matrix)
    return float(np.mean(report.f1[indices]))


def weighted_f1(cm: object, labels: object) -> float:
    """Support-weighted mean F1 over the supplied label indices."""
    matrix = _as_confusion(cm)
    indices = _validated_labels(matrix, labels)
    report = per_class_prf(matrix)
    weights = report.support[indices].astype(np.float64)
    total = float(weights.sum())
    if total <= 0:
        raise InvalidInputError("labels must include at least one supported class")
    return float(np.sum(report.f1[indices] * weights) / total)


def topk_accuracy(scores: object, y_true: object, k: int) -> float:
    """Fraction of examples whose true label is among the ``k`` best scores."""
    matrix = _as_scores(scores)
    true = _as_label_array(y_true, "y_true")
    if matrix.shape[0] != true.shape[0]:
        raise InvalidInputError("scores and y_true must have the same number of rows")
    if not 1 <= k <= matrix.shape[1]:
        raise InvalidInputError("k must be between 1 and the number of classes")
    if np.any(true < 0) or np.any(true >= matrix.shape[1]):
        raise InvalidInputError("y_true contains a label outside the label space")
    best = np.argsort(-matrix, axis=1, kind="stable")[:, :k]
    return float(np.mean(np.any(best == true[:, None], axis=1)))


def scenario_accuracy(y_true: object, y_pred: object, scenario_map: object) -> float:
    """Accuracy after mapping intent indices to their derived scenario (D1)."""
    true = _as_label_array(y_true, "y_true")
    predicted = _as_label_array(y_pred, "y_pred")
    if true.shape != predicted.shape:
        raise InvalidInputError("y_true and y_pred must have the same length")
    mapping = np.asarray(scenario_map)
    if mapping.ndim != 1 or mapping.size == 0:
        raise InvalidInputError("scenario_map must be a non-empty one-dimensional array")
    if np.any(true < 0) or np.any(true >= mapping.size):
        raise InvalidInputError("y_true contains an intent outside scenario_map")
    if np.any(predicted < 0) or np.any(predicted >= mapping.size):
        raise InvalidInputError("y_pred contains an intent outside scenario_map")
    return float(np.mean(mapping[true] == mapping[predicted]))


@dataclass(frozen=True, slots=True)
class ConfusionEntry:
    """One off-diagonal confusion count."""

    true: int
    predicted: int
    count: int


def top_confusions(cm: object, k: int) -> tuple[ConfusionEntry, ...]:
    """Most frequent off-diagonal confusions, ordered by count then indices."""
    matrix = _as_confusion(cm)
    if k <= 0:
        raise InvalidInputError("k must be positive")
    rows, columns = np.nonzero(matrix)
    candidates = [
        ConfusionEntry(true=int(row), predicted=int(column), count=int(matrix[row, column]))
        for row, column in zip(rows, columns, strict=True)
        if row != column
    ]
    candidates.sort(key=lambda entry: (-entry.count, entry.true, entry.predicted))
    return tuple(candidates[:k])
