"""Confusion-based metrics cross-checked against scikit-learn."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from stc.domain.errors import InvalidInputError
from stc.domain.metrics import (
    ConfusionEntry,
    PerClassMetrics,
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_prf,
    present_labels,
    scenario_accuracy,
    top_confusions,
    topk_accuracy,
    weighted_f1,
)

pytestmark = pytest.mark.unit

N_CLASSES = 6


def _pairs() -> st.SearchStrategy[list[tuple[int, int]]]:
    return st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=N_CLASSES - 1),
            st.integers(min_value=0, max_value=N_CLASSES - 1),
        ),
        min_size=1,
        max_size=80,
    )


@given(_pairs())
def test_confusion_matrix_counts_match_numpy(pairs: list[tuple[int, int]]) -> None:
    y_true = np.array([true for true, _ in pairs], dtype=np.int64)
    y_pred = np.array([pred for _, pred in pairs], dtype=np.int64)

    matrix = confusion_matrix(y_true, y_pred, N_CLASSES)

    assert matrix.shape == (N_CLASSES, N_CLASSES)
    assert matrix.dtype == np.int64
    assert matrix.sum() == len(pairs)
    for true, pred in set(pairs):
        expected = sum(1 for t, p in pairs if (t, p) == (true, pred))
        assert matrix[true, pred] == expected


@given(_pairs())
def test_accuracy_matches_sklearn(pairs: list[tuple[int, int]]) -> None:
    y_true = np.array([true for true, _ in pairs], dtype=np.int64)
    y_pred = np.array([pred for _, pred in pairs], dtype=np.int64)

    assert accuracy(y_true, y_pred) == pytest.approx(accuracy_score(y_true, y_pred), abs=1e-12)


def test_present_labels_are_sorted_and_supported() -> None:
    y_true = np.array([3, 1, 1, 3, 5], dtype=np.int64)

    labels = present_labels(y_true)

    assert labels.tolist() == [1, 3, 5]
    assert labels.dtype == np.int64


@given(_pairs())
def test_per_class_prf_matches_sklearn(pairs: list[tuple[int, int]]) -> None:
    y_true = np.array([true for true, _ in pairs], dtype=np.int64)
    y_pred = np.array([pred for _, pred in pairs], dtype=np.int64)
    matrix = confusion_matrix(y_true, y_pred, N_CLASSES)

    report = per_class_prf(matrix)

    assert isinstance(report, PerClassMetrics)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
        zero_division=0,
    )
    np.testing.assert_allclose(report.precision, precision, atol=1e-12)
    np.testing.assert_allclose(report.recall, recall, atol=1e-12)
    np.testing.assert_allclose(report.f1, f1, atol=1e-12)
    np.testing.assert_array_equal(report.support, support)


@given(_pairs())
def test_macro_f1_matches_sklearn_over_present_labels(pairs: list[tuple[int, int]]) -> None:
    y_true = np.array([true for true, _ in pairs], dtype=np.int64)
    y_pred = np.array([pred for _, pred in pairs], dtype=np.int64)
    matrix = confusion_matrix(y_true, y_pred, N_CLASSES)
    labels = present_labels(y_true)

    expected = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    assert macro_f1(matrix, labels) == pytest.approx(expected, abs=1e-12)


@given(_pairs())
def test_weighted_f1_matches_sklearn_over_present_labels(
    pairs: list[tuple[int, int]],
) -> None:
    y_true = np.array([true for true, _ in pairs], dtype=np.int64)
    y_pred = np.array([pred for _, pred in pairs], dtype=np.int64)
    matrix = confusion_matrix(y_true, y_pred, N_CLASSES)
    labels = present_labels(y_true)

    expected = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    assert weighted_f1(matrix, labels) == pytest.approx(expected, abs=1e-12)


@given(_pairs())
def test_macro_f1_is_a_probability(pairs: list[tuple[int, int]]) -> None:
    y_true = np.array([true for true, _ in pairs], dtype=np.int64)
    y_pred = np.array([pred for _, pred in pairs], dtype=np.int64)
    matrix = confusion_matrix(y_true, y_pred, N_CLASSES)

    score = macro_f1(matrix, present_labels(y_true))

    assert 0.0 <= score <= 1.0


def test_perfect_predictions_score_one() -> None:
    y_true = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    matrix = confusion_matrix(y_true, y_true, 3)

    assert accuracy(y_true, y_true) == 1.0
    assert macro_f1(matrix, present_labels(y_true)) == 1.0
    assert weighted_f1(matrix, present_labels(y_true)) == 1.0
    assert topk_accuracy(np.eye(3)[y_true], y_true, k=1) == 1.0


def test_macro_f1_is_invariant_to_label_relabelling() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 2, 2, 1, 1, 0, 2], dtype=np.int64)
    permutation = np.array([2, 0, 1], dtype=np.int64)

    original = macro_f1(confusion_matrix(y_true, y_pred, 3), present_labels(y_true))
    permuted = macro_f1(
        confusion_matrix(permutation[y_true], permutation[y_pred], 3),
        present_labels(permutation[y_true]),
    )

    assert original == pytest.approx(permuted, abs=1e-12)


def test_topk_accuracy_uses_the_best_scores() -> None:
    scores = np.array(
        [
            [9.0, 1.0, 0.0],
            [0.0, 9.0, 1.0],
            [0.0, 1.0, 9.0],
            [1.0, 9.0, 0.0],
        ]
    )
    y_true = np.array([0, 0, 2, 1], dtype=np.int64)

    assert topk_accuracy(scores, y_true, k=1) == pytest.approx(0.75)
    assert topk_accuracy(scores, y_true, k=2) == pytest.approx(0.75)
    assert topk_accuracy(scores, y_true, k=3) == 1.0


def test_scenario_accuracy_maps_intents_to_scenarios() -> None:
    scenario_map = np.array([0, 0, 1], dtype=np.int64)
    y_true = np.array([0, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 2, 2, 1], dtype=np.int64)

    # Predictions 1 and 2 are wrong at intent level, but 1 -> 0 is right at
    # scenario level while 2 -> 2 and 1 -> 2 are not.
    assert scenario_accuracy(y_true, y_pred, scenario_map) == pytest.approx(0.5)


def test_top_confusions_are_sorted_by_count_then_indices() -> None:
    matrix = np.zeros((4, 4), dtype=np.int64)
    matrix[0, 1] = 2
    matrix[2, 3] = 5
    matrix[1, 0] = 2
    matrix[0, 2] = 2
    matrix[3, 3] = 100

    result = top_confusions(matrix, k=3)

    assert result == (
        ConfusionEntry(true=2, predicted=3, count=5),
        ConfusionEntry(true=0, predicted=1, count=2),
        ConfusionEntry(true=0, predicted=2, count=2),
    )


def test_invalid_shapes_and_labels_are_rejected() -> None:
    with pytest.raises(InvalidInputError):
        confusion_matrix(np.array([0, 1]), np.array([0]), 2)
    with pytest.raises(InvalidInputError):
        confusion_matrix(np.array([0, 2]), np.array([0, 1]), 2)
    with pytest.raises(InvalidInputError):
        confusion_matrix(np.array([0.5, 1.0]), np.array([0, 1]), 2)
    with pytest.raises(InvalidInputError):
        confusion_matrix(np.zeros((2, 2)), np.array([0, 1]), 2)
    with pytest.raises(InvalidInputError):
        present_labels(np.array([[]], dtype=np.int64))
    with pytest.raises(InvalidInputError):
        per_class_prf(np.array([1, 2]))
    with pytest.raises(InvalidInputError):
        per_class_prf(np.array([[1, -1], [0, 1]]))
    with pytest.raises(InvalidInputError):
        macro_f1(np.eye(2, dtype=np.int64), np.array([], dtype=np.int64))
    with pytest.raises(InvalidInputError):
        macro_f1(np.eye(2, dtype=np.int64), np.array([5], dtype=np.int64))
    with pytest.raises(InvalidInputError):
        accuracy(np.array([0]), np.array([0, 1]))
    with pytest.raises(InvalidInputError):
        topk_accuracy(np.ones((2, 3)), np.array([0, 1]), k=0)
    with pytest.raises(InvalidInputError):
        topk_accuracy(np.ones((2, 3)), np.array([0, 1]), k=4)
    with pytest.raises(InvalidInputError):
        topk_accuracy(np.ones(3), np.array([0, 1]), k=1)
    with pytest.raises(InvalidInputError):
        scenario_accuracy(np.array([1]), np.array([0]), np.array([0], dtype=np.int64))
    with pytest.raises(InvalidInputError):
        top_confusions(np.eye(2, dtype=np.int64), k=0)
