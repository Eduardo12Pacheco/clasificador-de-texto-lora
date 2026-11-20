"""Bootstrap confidence intervals and exact McNemar tests."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import binomtest

from stc.domain.errors import InvalidInputError
from stc.domain.stats import (
    ConfidenceInterval,
    McNemarResult,
    bootstrap_ci,
    mcnemar_exact,
    paired_bootstrap_delta,
)

pytestmark = pytest.mark.unit


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


@pytest.fixture
def labels() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 5, size=200)
    y_pred = np.where(rng.random(200) < 0.7, y_true, rng.integers(0, 5, size=200))
    return y_true, y_pred


def test_bootstrap_ci_contains_the_point_estimate(labels: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = labels

    interval = bootstrap_ci(_accuracy, y_true, y_pred, B=500, seed=13)

    assert isinstance(interval, ConfidenceInterval)
    assert interval.estimate == pytest.approx(_accuracy(y_true, y_pred))
    assert interval.lower <= interval.estimate <= interval.upper
    assert 0.0 <= interval.lower <= 1.0
    assert 0.0 <= interval.upper <= 1.0


def test_bootstrap_is_reproducible_for_a_fixed_seed(labels: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = labels

    first = bootstrap_ci(_accuracy, y_true, y_pred, B=400, seed=13)
    second = bootstrap_ci(_accuracy, y_true, y_pred, B=400, seed=13)
    other = bootstrap_ci(_accuracy, y_true, y_pred, B=400, seed=99)

    assert first == second
    assert (first.lower, first.upper) != (other.lower, other.upper)


def test_paired_delta_uses_the_same_resamples() -> None:
    y_true = np.array([0, 1, 2, 3] * 50, dtype=np.int64)
    same = y_true.copy()

    interval = paired_bootstrap_delta(_accuracy, y_true, same, same, B=200, seed=13)

    assert interval.estimate == 0.0
    assert interval.lower == 0.0
    assert interval.upper == 0.0


@pytest.mark.parametrize(("b", "c"), [(0, 0), (1, 0), (0, 1), (5, 5), (10, 3), (30, 25)])
def test_mcnemar_matches_scipy_exact_binomial(b: int, c: int) -> None:
    base = np.zeros(100, dtype=bool)
    correct_a = base.copy()
    correct_b = base.copy()
    correct_a[:b] = True
    correct_b[100 - c :] = True

    result = mcnemar_exact(correct_a, correct_b)

    assert isinstance(result, McNemarResult)
    assert (result.b, result.c) == (b, c)
    n = b + c
    expected = 1.0 if n == 0 else binomtest(min(b, c), n, 0.5).pvalue
    assert result.p == pytest.approx(min(1.0, expected), abs=1e-12)


@given(
    st.integers(min_value=0, max_value=40),
    st.integers(min_value=0, max_value=40),
    st.integers(min_value=0, max_value=10),
)
def test_mcnemar_properties(b: int, c: int, both: int) -> None:
    n = b + c + both
    correct_a = np.zeros(n, dtype=bool)
    correct_b = np.zeros(n, dtype=bool)
    correct_a[: both + b] = True
    correct_b[:both] = True
    correct_b[both + b : both + b + c] = True

    result = mcnemar_exact(correct_a, correct_b)

    assert 0.0 <= result.p <= 1.0
    if b == c:
        assert result.p == 1.0
    assert (result.b, result.c) == (b, c)


def test_invalid_inputs_are_rejected() -> None:
    y_true = np.array([0, 1, 2], dtype=np.int64)
    with pytest.raises(InvalidInputError):
        bootstrap_ci(_accuracy, y_true, y_true[:2])
    with pytest.raises(InvalidInputError):
        bootstrap_ci(_accuracy, y_true, y_true, B=0)
    with pytest.raises(InvalidInputError):
        bootstrap_ci(_accuracy, y_true, y_true, alpha=0.0)
    with pytest.raises(InvalidInputError):
        bootstrap_ci(_accuracy, y_true, y_true, alpha=1.0)
    with pytest.raises(InvalidInputError):
        paired_bootstrap_delta(_accuracy, y_true, y_true, y_true[:2])
    with pytest.raises(InvalidInputError):
        paired_bootstrap_delta(_accuracy, y_true, y_true, y_true, B=0)
    with pytest.raises(InvalidInputError):
        mcnemar_exact(np.array([True]), np.array([True, False]))
    with pytest.raises(InvalidInputError):
        mcnemar_exact(np.array([1, 0]), np.array([1.5, 0.0]))
