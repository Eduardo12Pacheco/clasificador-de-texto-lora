"""Calibration metrics, temperature scaling and the review threshold."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import softmax as scipy_softmax

from stc.domain.calibration import (
    Calibration,
    ReliabilityBins,
    SelectiveMetrics,
    brier,
    choose_review_threshold,
    ece,
    fit_temperature,
    nll,
    reliability_bins,
    selective_metrics,
    softmax,
)
from stc.domain.errors import InvalidInputError

pytestmark = pytest.mark.unit


def test_softmax_rows_sum_to_one() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=np.float64)

    probs = softmax(logits)

    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-12)
    np.testing.assert_allclose(probs, scipy_softmax(logits, axis=1), atol=1e-12)


def test_softmax_1d_and_temperature_match_scipy() -> None:
    logits = np.array([0.5, -1.0, 2.0], dtype=np.float64)

    np.testing.assert_allclose(softmax(logits, T=2.0), scipy_softmax(logits / 2.0), atol=1e-12)


@given(
    st.integers(min_value=1, max_value=8),
    st.lists(
        st.integers(min_value=-20, max_value=20),
        min_size=1,
        max_size=10,
    ),
    st.floats(min_value=0.05, max_value=20.0),
)
def test_temperature_scaling_never_changes_argmax(
    n_classes: int, values: list[int], temperature: float
) -> None:
    logits = np.resize(np.array(values, dtype=np.float64), n_classes)

    assert int(np.argmax(logits)) == int(np.argmax(softmax(logits, T=temperature)))


def test_softmax_rejects_bad_temperature_and_shape() -> None:
    with pytest.raises(InvalidInputError):
        softmax(np.ones((2, 3)), T=0.0)
    with pytest.raises(InvalidInputError):
        softmax(np.ones((2, 3)), T=-1.0)
    with pytest.raises(InvalidInputError):
        softmax(np.ones((2, 3, 4)))


def test_nll_and_brier_are_correct() -> None:
    probs = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
    y = np.array([0, 1], dtype=np.int64)

    assert nll(probs, y) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)
    assert brier(probs, y) == pytest.approx(((0.2**2 + 0.2**2) + (0.3**2 + 0.3**2)) / 2)


def test_nll_and_brier_reject_invalid_input() -> None:
    probs = np.array([[0.8, 0.2]], dtype=np.float64)
    with pytest.raises(InvalidInputError):
        nll(probs, np.array([2], dtype=np.int64))
    with pytest.raises(InvalidInputError):
        nll(probs, np.array([0, 1], dtype=np.int64))
    with pytest.raises(InvalidInputError):
        brier(np.array([0.5, 0.5]), np.array([0], dtype=np.int64))
    with pytest.raises(InvalidInputError):
        brier(probs, np.array([[0]], dtype=np.int64))


def test_reliability_bins_cover_every_example() -> None:
    confidence = np.array([0.05, 0.15, 0.33, 0.67, 0.95, 1.0], dtype=np.float64)
    correct = np.array([False, False, True, True, True, True])

    bins = reliability_bins(confidence, correct, n_bins=15)

    assert isinstance(bins, ReliabilityBins)
    assert bins.counts.sum() == len(confidence)
    assert bins.bin_lower.shape == bins.bin_upper.shape == bins.counts.shape == (15,)
    assert bins.ece == pytest.approx(ece(confidence, correct, n_bins=15))


def test_ece_is_zero_for_perfectly_calibrated_synthetic_data() -> None:
    n_bins = 15
    n = 20_000
    per_bin = n // n_bins
    confidence: list[float] = []
    correct: list[bool] = []
    for index in range(n_bins):
        midpoint = (index + 0.5) / n_bins
        n_correct = round(midpoint * per_bin)
        confidence.extend([midpoint] * per_bin)
        correct.extend([True] * n_correct + [False] * (per_bin - n_correct))

    value = ece(np.array(confidence), np.array(correct), n_bins=n_bins)

    assert 0.0 <= value <= 0.02


def test_ece_is_zero_for_an_empty_input() -> None:
    assert ece(np.array([]), np.array([], dtype=bool)) == 0.0


def test_ece_is_a_probability() -> None:
    rng = np.random.default_rng(3)
    confidence = rng.random(500)
    correct = rng.random(500) < confidence

    assert 0.0 <= ece(confidence, correct) <= 1.0


def _synthetic_calibration_data(
    temperature: float, n: int = 20_000
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    true_logits = rng.normal(size=(n, 5))
    y = np.array([rng.choice(5, p=scipy_softmax(row)) for row in true_logits], dtype=np.int64)
    return true_logits * temperature, y


def test_fit_temperature_recovers_a_known_scale() -> None:
    logits, y = _synthetic_calibration_data(temperature=2.0)

    fitted = fit_temperature(logits, y)

    assert fitted == pytest.approx(2.0, rel=0.05)


def test_fit_temperature_recovers_the_identity() -> None:
    logits, y = _synthetic_calibration_data(temperature=1.0)

    assert fit_temperature(logits, y) == pytest.approx(1.0, rel=0.05)


def test_fit_temperature_rejects_empty_and_mismatched_input() -> None:
    logits = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(InvalidInputError):
        fit_temperature(logits, np.array([], dtype=np.int64))
    with pytest.raises(InvalidInputError):
        fit_temperature(logits, np.array([0], dtype=np.int64))
    with pytest.raises(InvalidInputError):
        fit_temperature(np.zeros(3), np.array([0], dtype=np.int64))


def test_choose_review_threshold_picks_the_smallest_qualifying_cutoff() -> None:
    confidence = np.array([0.99] * 90 + [0.10] * 10, dtype=np.float64)
    correct = np.array([True] * 90 + [False] * 10)

    threshold = choose_review_threshold(confidence, correct, target=0.95, min_coverage=0.20)

    assert threshold == pytest.approx(0.99)


def test_choose_review_threshold_respects_minimum_coverage() -> None:
    # The perfect group covers only 10 % of dev, so the rule must ignore it and
    # fall back to a lower cut-off that meets the 20 % coverage floor.
    confidence = np.array([0.99] * 10 + [0.80] * 90, dtype=np.float64)
    correct = np.array([True] * 100)

    threshold = choose_review_threshold(confidence, correct, target=0.95, min_coverage=0.20)

    assert threshold == pytest.approx(0.80)


def test_choose_review_threshold_falls_back_to_one() -> None:
    confidence = np.array([0.9, 0.8, 0.7], dtype=np.float64)
    correct = np.array([False, False, False])

    assert choose_review_threshold(confidence, correct) == 1.0


def test_selective_metrics() -> None:
    confidence = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float64)
    correct = np.array([True, False, True, False])

    metrics = selective_metrics(confidence, correct, tau=0.75)

    assert isinstance(metrics, SelectiveMetrics)
    assert metrics.coverage == pytest.approx(0.5)
    assert metrics.selective_accuracy == pytest.approx(0.5)
    assert metrics.n_selected == 2


def test_selective_metrics_with_an_empty_selection() -> None:
    metrics = selective_metrics(np.array([0.5, 0.6]), np.array([True, False]), tau=0.9)

    assert metrics.coverage == 0.0
    assert metrics.selective_accuracy == 0.0
    assert metrics.n_selected == 0


def test_selection_helpers_reject_invalid_input() -> None:
    confidence = np.array([0.5, 0.6], dtype=np.float64)
    correct = np.array([True, False])
    with pytest.raises(InvalidInputError):
        choose_review_threshold(confidence, correct, target=1.5)
    with pytest.raises(InvalidInputError):
        choose_review_threshold(confidence, correct, min_coverage=1.5)
    with pytest.raises(InvalidInputError):
        choose_review_threshold(confidence, correct[:1])
    with pytest.raises(InvalidInputError):
        choose_review_threshold(np.array([1.2, 0.5]), correct)
    with pytest.raises(InvalidInputError):
        selective_metrics(confidence, correct, tau=-0.1)


def test_calibration_value_object_round_trips() -> None:
    calibration = Calibration(temperature=1.5, review_threshold=0.7)

    assert calibration.to_dict() == {"review_threshold": 0.7, "temperature": 1.5}
    assert Calibration.from_dict(calibration.to_dict()) == calibration
    with pytest.raises(InvalidInputError):
        Calibration(temperature=0.0, review_threshold=0.5)
    with pytest.raises(InvalidInputError):
        Calibration.from_dict({"temperature": "x", "review_threshold": 0.5})
