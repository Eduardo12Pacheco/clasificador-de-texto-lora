"""Bootstrap confidence intervals and the exact McNemar test (D2)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stc.domain.errors import InvalidInputError

__all__ = [
    "ConfidenceInterval",
    "McNemarResult",
    "bootstrap_ci",
    "mcnemar_exact",
    "paired_bootstrap_delta",
]

MetricFunction = Callable[[NDArray[np.int64], NDArray[np.int64]], float]


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A point estimate and its percentile-bootstrap interval."""

    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Discordant pair counts and the exact two-sided p-value."""

    b: int
    c: int
    p: float


def _as_label_array(values: object, name: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise InvalidInputError(f"{name} must be a non-empty one-dimensional array")
    if not np.issubdtype(array.dtype, np.integer):
        raise InvalidInputError(f"{name} must contain integer labels")
    return np.asarray(array, dtype=np.int64)


def _validate_bootstrap(B: int, alpha: float) -> None:
    if B <= 0:
        raise InvalidInputError("B must be positive")
    if not 0.0 < alpha < 1.0:
        raise InvalidInputError("alpha must be between 0 and 1")


def _percentile_interval(
    estimate: float, values: NDArray[np.float64], alpha: float
) -> ConfidenceInterval:
    lower, upper = np.percentile(values, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return ConfidenceInterval(estimate=estimate, lower=float(lower), upper=float(upper))


def bootstrap_ci(
    metric_fn: MetricFunction,
    y_true: object,
    y_pred: object,
    *,
    B: int = 2000,
    seed: int = 13,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """Percentile-bootstrap ``metric_fn`` with a fixed resampling seed (D2)."""
    _validate_bootstrap(B, alpha)
    true = _as_label_array(y_true, "y_true")
    predicted = _as_label_array(y_pred, "y_pred")
    if true.shape != predicted.shape:
        raise InvalidInputError("y_true and y_pred must have the same length")

    rng = np.random.default_rng(seed)
    n = true.shape[0]
    values = np.empty(B, dtype=np.float64)
    for index in range(B):
        sample = rng.integers(0, n, size=n)
        values[index] = metric_fn(true[sample], predicted[sample])
    estimate = float(metric_fn(true, predicted))
    return _percentile_interval(estimate, values, alpha)


def paired_bootstrap_delta(
    metric_fn: MetricFunction,
    y_true: object,
    y_pred_a: object,
    y_pred_b: object,
    *,
    B: int = 2000,
    seed: int = 13,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """Bootstrap ``metric(a) - metric(b)`` with shared resample indices (D2)."""
    _validate_bootstrap(B, alpha)
    true = _as_label_array(y_true, "y_true")
    predicted_a = _as_label_array(y_pred_a, "y_pred_a")
    predicted_b = _as_label_array(y_pred_b, "y_pred_b")
    if not (true.shape == predicted_a.shape == predicted_b.shape):
        raise InvalidInputError("y_true and both prediction arrays must have the same length")

    rng = np.random.default_rng(seed)
    n = true.shape[0]
    values = np.empty(B, dtype=np.float64)
    for index in range(B):
        sample = rng.integers(0, n, size=n)
        values[index] = metric_fn(true[sample], predicted_a[sample]) - metric_fn(
            true[sample], predicted_b[sample]
        )
    estimate = float(metric_fn(true, predicted_a) - metric_fn(true, predicted_b))
    return _percentile_interval(estimate, values, alpha)


def _as_boolean_array(values: object, name: str) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise InvalidInputError(f"{name} must be a one-dimensional array")
    if array.dtype != np.bool_ and (
        not np.issubdtype(array.dtype, np.integer) or np.any((array != 0) & (array != 1))
    ):
        raise InvalidInputError(f"{name} must be boolean")
    return np.asarray(array, dtype=np.bool_)


def mcnemar_exact(correct_a: object, correct_b: object) -> McNemarResult:
    """Exact two-sided McNemar test over paired per-example correctness (D2)."""
    a = _as_boolean_array(correct_a, "correct_a")
    b_array = _as_boolean_array(correct_b, "correct_b")
    if a.shape != b_array.shape:
        raise InvalidInputError("correct_a and correct_b must have the same length")

    b = int(np.sum(a & ~b_array))
    c = int(np.sum(~a & b_array))
    n = b + c
    if n == 0:
        return McNemarResult(b=0, c=0, p=1.0)
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    p_value = min(1.0, 2.0 * tail / (2**n))
    return McNemarResult(b=b, c=c, p=float(p_value))
