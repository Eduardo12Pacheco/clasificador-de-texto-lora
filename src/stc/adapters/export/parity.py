"""ONNX parity and the pre-registered quantisation gate (S11/D12)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stc.domain.metrics import confusion_matrix, macro_f1, present_labels

__all__ = [
    "GATE_AGREEMENT",
    "GATE_MACRO_F1_DROP",
    "GateResult",
    "compare_logits",
    "quantization_gate",
]

GATE_MACRO_F1_DROP = 0.005
GATE_AGREEMENT = 0.99


@dataclass(frozen=True, slots=True)
class GateResult:
    """Outcome of the D12 gate for one quantisation variant."""

    passed: bool
    macro_f1_drop: float
    agreement: float
    macro_f1_fp32: float
    macro_f1_variant: float

    def to_dict(self) -> dict[str, object]:
        """Return the JSON row written to ``quantization_dev.json``."""
        return {
            "passed": self.passed,
            "gate_passed": self.passed,
            "macro_f1_drop": self.macro_f1_drop,
            "agreement": self.agreement,
            "macro_f1_fp32": self.macro_f1_fp32,
            "macro_f1_variant": self.macro_f1_variant,
        }


def compare_logits(
    torch_scores: object,
    onnx_scores: object,
    *,
    max_logit_diff: float = 5e-3,
) -> dict[str, object]:
    """Compare torch and ONNX logits on the same dev examples."""
    left = np.asarray(torch_scores, dtype=np.float64)
    right = np.asarray(onnx_scores, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"score shapes differ: {left.shape} != {right.shape}")
    difference = float(np.max(np.abs(left - right))) if left.size else 0.0
    agreement = float(np.mean(np.argmax(left, axis=1) == np.argmax(right, axis=1)))
    return {
        "max_abs_logit_diff": difference,
        "agreement": agreement,
        "max_allowed_logit_diff": max_logit_diff,
        "passed": bool(difference <= max_logit_diff and agreement == 1.0),
    }


def _macro_f1(scores: NDArray[np.float64], y_true: NDArray[np.int64], n_labels: int) -> float:
    predicted = np.argmax(scores, axis=1).astype(np.int64)
    matrix = confusion_matrix(y_true, predicted, n_labels)
    return macro_f1(matrix, present_labels(y_true))


def quantization_gate(
    fp32_scores: object,
    variant_scores: object,
    y_true: Sequence[int] | object,
) -> GateResult:
    """Apply the binding gate: macro-F1 drop <= 0.5 pt and agreement >= 99 %."""
    reference = np.asarray(fp32_scores, dtype=np.float64)
    variant = np.asarray(variant_scores, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)
    if reference.shape != variant.shape:
        raise ValueError(f"score shapes differ: {reference.shape} != {variant.shape}")
    if labels.shape != (reference.shape[0],):
        raise ValueError("y_true must have one label per score row")
    n_labels = reference.shape[1]
    fp32_f1 = _macro_f1(reference, labels, n_labels)
    variant_f1 = _macro_f1(variant, labels, n_labels)
    drop = fp32_f1 - variant_f1
    agreement = float(np.mean(np.argmax(reference, axis=1) == np.argmax(variant, axis=1)))
    passed = bool(drop <= GATE_MACRO_F1_DROP + 1e-12 and agreement >= GATE_AGREEMENT)
    return GateResult(
        passed=passed,
        macro_f1_drop=float(drop),
        agreement=agreement,
        macro_f1_fp32=float(fp32_f1),
        macro_f1_variant=float(variant_f1),
    )
