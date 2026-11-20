"""Calibration, temperature scaling and the review threshold (D10)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stc.domain.errors import InvalidInputError

__all__ = [
    "Calibration",
    "ReliabilityBins",
    "SelectiveMetrics",
    "brier",
    "choose_review_threshold",
    "ece",
    "fit_temperature",
    "nll",
    "reliability_bins",
    "selective_metrics",
    "softmax",
]

_MIN_TEMPERATURE = 0.05
_MAX_TEMPERATURE = 20.0
_GOLDEN_SECTION_ITERATIONS = 80
_GOLDEN_RATIO = (math.sqrt(5.0) - 1.0) / 2.0


def _as_float_array(values: object, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim == 0:
        raise InvalidInputError(f"{name} must be an array")
    if not np.issubdtype(array.dtype, np.number):
        raise InvalidInputError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise InvalidInputError(f"{name} must be finite")
    return np.asarray(array, dtype=np.float64)


def _as_label_array(values: object, name: str, n_rows: int) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size != n_rows:
        raise InvalidInputError(f"{name} must have one integer label per row")
    if not np.issubdtype(array.dtype, np.integer):
        raise InvalidInputError(f"{name} must contain integer labels")
    return np.asarray(array, dtype=np.int64)


def _as_confidence(values: object, name: str) -> NDArray[np.float64]:
    array = _as_float_array(values, name)
    if array.ndim != 1:
        raise InvalidInputError(f"{name} must be one-dimensional")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise InvalidInputError(f"{name} must lie in [0, 1]")
    return array


def _as_correct(values: object, name: str) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise InvalidInputError(f"{name} must be one-dimensional")
    if array.dtype != np.bool_:
        raise InvalidInputError(f"{name} must be boolean")
    return np.asarray(array, dtype=np.bool_)


def softmax(z: object, T: float = 1.0) -> NDArray[np.float64]:
    """Numerically stable ``softmax(z / T)`` for one row or a batch of rows."""
    if not np.isfinite(T) or T <= 0.0:
        raise InvalidInputError("temperature must be positive and finite")
    logits = _as_float_array(z, "logits")
    if logits.ndim not in (1, 2):
        raise InvalidInputError("logits must be one- or two-dimensional")
    if logits.size == 0:
        return np.empty_like(logits)
    scaled = logits / T
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def _validate_probabilities(
    probs: object, y: object
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    matrix = _as_float_array(probs, "probs")
    if matrix.ndim != 2:
        raise InvalidInputError("probs must be two-dimensional")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise InvalidInputError("probs must lie in [0, 1]")
    labels = _as_label_array(y, "y", matrix.shape[0])
    if np.any(labels < 0) or np.any(labels >= matrix.shape[1]):
        raise InvalidInputError("y contains a label outside the probability columns")
    return matrix, labels


def nll(probs: object, y: object) -> float:
    """Mean negative log-likelihood of the true labels."""
    matrix, labels = _validate_probabilities(probs, y)
    if matrix.shape[0] == 0:
        raise InvalidInputError("nll needs at least one example")
    chosen = matrix[np.arange(matrix.shape[0]), labels]
    tiny = np.finfo(np.float64).tiny
    return float(-np.mean(np.log(np.clip(chosen, tiny, 1.0))))


def brier(probs: object, y: object) -> float:
    """Mean squared error between probabilities and one-hot labels."""
    matrix, labels = _validate_probabilities(probs, y)
    if matrix.shape[0] == 0:
        raise InvalidInputError("brier needs at least one example")
    one_hot = np.zeros_like(matrix)
    one_hot[np.arange(matrix.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1)))


@dataclass(frozen=True, slots=True)
class ReliabilityBins:
    """Equal-width reliability bins over max-probability and correctness."""

    bin_lower: NDArray[np.float64]
    bin_upper: NDArray[np.float64]
    counts: NDArray[np.int64]
    confidence: NDArray[np.float64]
    accuracy: NDArray[np.float64]
    ece: float


def reliability_bins(confidence: object, correct: object, n_bins: int = 15) -> ReliabilityBins:
    """Bin max-probability and correctness into ``n_bins`` equal-width bins (D10)."""
    if n_bins <= 0:
        raise InvalidInputError("n_bins must be positive")
    values = _as_confidence(confidence, "confidence")
    outcomes = _as_correct(correct, "correct")
    if values.shape != outcomes.shape:
        raise InvalidInputError("confidence and correct must have the same length")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    bin_confidence = np.zeros(n_bins, dtype=np.float64)
    bin_accuracy = np.zeros(n_bins, dtype=np.float64)
    if values.size:
        indices = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)
        for index in range(n_bins):
            mask = indices == index
            count = int(np.sum(mask))
            counts[index] = count
            if count:
                bin_confidence[index] = float(np.mean(values[mask]))
                bin_accuracy[index] = float(np.mean(outcomes[mask]))
    gaps = np.abs(bin_accuracy - bin_confidence) * counts
    total = int(counts.sum())
    ece_value = float(gaps.sum() / total) if total else 0.0
    return ReliabilityBins(
        bin_lower=edges[:-1].copy(),
        bin_upper=edges[1:].copy(),
        counts=counts,
        confidence=bin_confidence,
        accuracy=bin_accuracy,
        ece=ece_value,
    )


def ece(confidence: object, correct: object, n_bins: int = 15) -> float:
    """Expected calibration error over max-probability bins (D10)."""
    return reliability_bins(confidence, correct, n_bins=n_bins).ece


def _nll_at_log_temperature(
    log_temperature: float, logits: NDArray[np.float64], y: NDArray[np.int64]
) -> float:
    return nll(softmax(logits, T=math.exp(log_temperature)), y)


def fit_temperature(logits: object, y: object) -> float:
    """Fit a single temperature by minimising NLL with golden-section search (D10)."""
    matrix = _as_float_array(logits, "logits")
    if matrix.ndim != 2:
        raise InvalidInputError("logits must be two-dimensional")
    if matrix.shape[0] == 0:
        raise InvalidInputError("logits must not be empty")
    labels = _as_label_array(y, "y", matrix.shape[0])
    if np.any(labels < 0) or np.any(labels >= matrix.shape[1]):
        raise InvalidInputError("y contains a label outside the logit columns")

    low = math.log(_MIN_TEMPERATURE)
    high = math.log(_MAX_TEMPERATURE)
    left = high - _GOLDEN_RATIO * (high - low)
    right = low + _GOLDEN_RATIO * (high - low)
    f_left = _nll_at_log_temperature(left, matrix, labels)
    f_right = _nll_at_log_temperature(right, matrix, labels)
    for _ in range(_GOLDEN_SECTION_ITERATIONS):
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - _GOLDEN_RATIO * (high - low)
            f_left = _nll_at_log_temperature(left, matrix, labels)
        else:
            low, left, f_left = left, right, f_right
            right = low + _GOLDEN_RATIO * (high - low)
            f_right = _nll_at_log_temperature(right, matrix, labels)
    return float(math.exp((low + high) / 2.0))


@dataclass(frozen=True, slots=True)
class SelectiveMetrics:
    """Coverage and accuracy of the examples above the review threshold."""

    coverage: float
    selective_accuracy: float
    n_selected: int


def choose_review_threshold(
    confidence: object,
    correct: object,
    target: float = 0.95,
    min_coverage: float = 0.20,
) -> float:
    """Smallest cut-off meeting target selective accuracy and coverage (D10)."""
    if not 0.0 < target <= 1.0:
        raise InvalidInputError("target must lie in (0, 1]")
    if not 0.0 < min_coverage <= 1.0:
        raise InvalidInputError("min_coverage must lie in (0, 1]")
    values = _as_confidence(confidence, "confidence")
    outcomes = _as_correct(correct, "correct")
    if values.shape != outcomes.shape:
        raise InvalidInputError("confidence and correct must have the same length")
    if values.size == 0:
        return 1.0

    for threshold in np.unique(values):
        selected = values >= threshold
        coverage = float(np.mean(selected))
        if coverage < min_coverage:
            continue
        selective_accuracy = float(np.mean(outcomes[selected]))
        if selective_accuracy >= target:
            return float(threshold)
    return 1.0


def selective_metrics(confidence: object, correct: object, tau: float) -> SelectiveMetrics:
    """Coverage and selective accuracy at a fixed review threshold."""
    if not 0.0 <= tau <= 1.0:
        raise InvalidInputError("tau must lie in [0, 1]")
    values = _as_confidence(confidence, "confidence")
    outcomes = _as_correct(correct, "correct")
    if values.shape != outcomes.shape:
        raise InvalidInputError("confidence and correct must have the same length")
    selected = values >= tau
    n_selected = int(np.sum(selected))
    coverage = float(np.mean(selected)) if values.size else 0.0
    selective_accuracy = float(np.mean(outcomes[selected])) if n_selected else 0.0
    return SelectiveMetrics(
        coverage=coverage,
        selective_accuracy=selective_accuracy,
        n_selected=n_selected,
    )


@dataclass(frozen=True, slots=True)
class Calibration:
    """Fitted temperature and review threshold for one classifier."""

    temperature: float
    review_threshold: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise InvalidInputError("temperature must be positive and finite")
        if not 0.0 <= self.review_threshold <= 1.0:
            raise InvalidInputError("review_threshold must lie in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation with stable keys."""
        return {
            "review_threshold": self.review_threshold,
            "temperature": self.temperature,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Calibration:
        """Rebuild a calibration object, rejecting malformed persisted values."""
        temperature = payload.get("temperature")
        review_threshold = payload.get("review_threshold")
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            raise InvalidInputError("temperature must be a number")
        if isinstance(review_threshold, bool) or not isinstance(review_threshold, int | float):
            raise InvalidInputError("review_threshold must be a number")
        return cls(
            temperature=float(temperature),
            review_threshold=float(review_threshold),
        )
