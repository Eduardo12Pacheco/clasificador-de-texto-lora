"""The D6 selection rule is table-driven and never picks Qwen."""

from __future__ import annotations

from pathlib import Path

import pytest

from stc.application.dto import ScreeningRun
from stc.application.services.backbone_selection import (
    BERTIN,
    MRBERT,
    QWEN,
    select_backbone,
)
from stc.domain.errors import InvalidInputError


def _run(backbone: str, score: float, seed: int = 13) -> ScreeningRun:
    return ScreeningRun(
        backbone=backbone,
        seed=seed,
        dev_macro_f1=score,
        dev_accuracy=score,
        trainable_params=100,
        epoch_seconds=1.0,
        peak_vram_mib=10.0,
        cpu_latency_ms_p50=5.0,
        exportable=backbone != QWEN,
        run_dir=Path(backbone.replace("/", "-")),
    )


def test_within_one_point_prefers_mrbert() -> None:
    selection = select_backbone([_run(MRBERT, 0.84), _run(BERTIN, 0.845)])

    assert selection.selected == MRBERT
    assert selection.rule_branch == "within_1pt_prefer_mrbert"


def test_bertin_more_than_one_point_ahead_wins() -> None:
    selection = select_backbone([_run(MRBERT, 0.82), _run(BERTIN, 0.84)])

    assert selection.selected == BERTIN
    assert selection.rule_branch == "bertin_ahead"


def test_mrbert_ahead_branch() -> None:
    selection = select_backbone([_run(MRBERT, 0.86), _run(BERTIN, 0.84)])

    assert selection.selected == MRBERT
    assert selection.rule_branch == "mrbert_ahead"


def test_qwen_is_never_selected_even_when_best() -> None:
    selection = select_backbone(
        [
            _run(MRBERT, 0.80),
            _run(BERTIN, 0.79),
            _run(QWEN, 0.99),
        ]
    )

    assert selection.selected == MRBERT


def test_only_qwen_is_an_error() -> None:
    with pytest.raises(InvalidInputError):
        select_backbone([_run(QWEN, 0.99)])


def test_means_are_per_backbone_across_seeds() -> None:
    selection = select_backbone(
        [_run(MRBERT, 0.80, 13), _run(MRBERT, 0.84, 42), _run(BERTIN, 0.81, 13)]
    )

    assert selection.per_backbone_means[MRBERT] == pytest.approx(0.82)
