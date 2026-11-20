"""Matplotlib figures for the evaluation reports (optional ``train`` extra)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from stc.domain.calibration import ReliabilityBins
from stc.domain.errors import MissingExtraError

__all__ = ["plot_confusion_matrix", "plot_per_class_f1", "plot_reliability_diagram"]

_MISSING_MATPLOTLIB = (
    "plotting needs the 'train' extra: uv tool install 'spanish-text-classifier-lora[train]'"
)


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise MissingExtraError(_MISSING_MATPLOTLIB) from error
    return plt


def _save(plt: Any, figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_reliability_diagram(bins: ReliabilityBins, path: Path) -> None:
    """Plot predicted confidence against observed accuracy per bin."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(4.0, 4.0))
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1, label="perfect")
    populated = bins.counts > 0
    centres = (bins.bin_lower[populated] + bins.bin_upper[populated]) / 2.0
    axis.plot(
        centres,
        bins.accuracy[populated],
        marker="o",
        color="#4f46e5",
        label="observed",
    )
    axis.set_xlabel("confidence")
    axis.set_ylabel("accuracy")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_title(f"Reliability (ECE={bins.ece:.3f})")
    axis.legend(loc="upper left", frameon=False)
    _save(plt, figure, path)


def plot_confusion_matrix(
    cm: NDArray[np.int64] | Sequence[Sequence[int]],
    intents: Sequence[str],
    path: Path,
) -> None:
    """Plot the row-normalised intent confusion matrix."""
    matrix = np.asarray(cm, dtype=np.float64)
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    plt = _pyplot()
    size = max(6.0, len(intents) * 0.16)
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xlabel("predicted intent")
    axis.set_ylabel("true intent")
    axis.set_title("Normalised confusion matrix")
    _save(plt, figure, path)


def plot_per_class_f1(
    intents: Sequence[str],
    f1_scores: Sequence[float],
    path: Path,
) -> None:
    """Plot per-intent F1 as a horizontal bar chart."""
    plt = _pyplot()
    height = max(4.0, len(intents) * 0.16)
    figure, axis = plt.subplots(figsize=(7.0, height))
    positions = np.arange(len(intents))
    axis.barh(positions, np.asarray(f1_scores, dtype=np.float64), color="#4f46e5")
    axis.set_yticks(positions)
    axis.set_yticklabels(list(intents), fontsize=6)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("F1")
    axis.set_title("Per-intent F1")
    _save(plt, figure, path)
