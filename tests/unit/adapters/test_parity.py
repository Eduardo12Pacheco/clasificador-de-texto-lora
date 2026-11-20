"""The D12 gate is a pure, pre-registered rule over logits."""

from __future__ import annotations

import numpy as np
import pytest

from stc.adapters.export.parity import (
    GATE_AGREEMENT,
    GATE_MACRO_F1_DROP,
    compare_logits,
    quantization_gate,
)


def _scores(n: int, n_labels: int = 2) -> np.ndarray:
    labels = np.arange(n) % n_labels
    scores = np.full((n, n_labels), 0.1, dtype=np.float64)
    scores[np.arange(n), labels] = 5.0
    return scores


def test_compare_logits_passes_exact_parity() -> None:
    scores = _scores(10)
    report = compare_logits(scores, scores)
    assert report["passed"] is True
    assert report["agreement"] == 1.0
    assert report["max_abs_logit_diff"] == 0.0


def test_compare_logits_rejects_different_shapes() -> None:
    with pytest.raises(ValueError):
        compare_logits(np.zeros((2, 3)), np.zeros((3, 3)))


def test_gate_passes_a_tiny_drop_and_high_agreement() -> None:
    y_true = np.arange(300) % 2
    reference = _scores(300)
    variant = reference.copy()
    variant[0] = variant[0][::-1]  # one flipped prediction out of 300
    result = quantization_gate(reference, variant, y_true)
    assert result.agreement >= GATE_AGREEMENT
    assert result.macro_f1_drop <= GATE_MACRO_F1_DROP
    assert result.passed is True


def test_gate_fails_on_a_macro_f1_drop() -> None:
    y_true = np.arange(200) % 2
    reference = _scores(200)
    variant = reference.copy()
    for index in (0, 2, 4):
        variant[index] = variant[index][::-1]
    result = quantization_gate(reference, variant, y_true)
    assert result.macro_f1_drop > GATE_MACRO_F1_DROP
    assert result.passed is False


def test_gate_fails_on_agreement() -> None:
    y_true = np.arange(300) % 2
    reference = _scores(300)
    variant = reference.copy()
    for index in range(0, 9):
        variant[index] = variant[index][::-1]
    result = quantization_gate(reference, variant, y_true)
    assert result.agreement < GATE_AGREEMENT
    assert result.passed is False


def test_gate_rejects_shape_mismatches() -> None:
    with pytest.raises(ValueError):
        quantization_gate(np.zeros((2, 3)), np.zeros((2, 3)), [0])
    with pytest.raises(ValueError):
        quantization_gate(np.zeros((2, 3)), np.zeros((3, 3)), [0, 1])
