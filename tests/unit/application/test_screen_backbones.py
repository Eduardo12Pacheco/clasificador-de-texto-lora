"""The screening use case writes both contract files from a runner."""

from __future__ import annotations

import json
from pathlib import Path

from stc.adapters.reporting.json_writer import JsonReportWriter
from stc.application.dto import ScreeningRun
from stc.application.services.backbone_selection import BERTIN, MRBERT
from stc.application.use_cases.screen_backbones import ScreenBackbones


class _FakeRunner:
    def run(self):
        return [
            ScreeningRun(
                backbone=MRBERT,
                seed=13,
                dev_macro_f1=0.84,
                dev_accuracy=0.85,
                trainable_params=100,
                epoch_seconds=1.0,
                peak_vram_mib=10.0,
                cpu_latency_ms_p50=5.0,
                exportable=True,
                run_dir=Path("a"),
            ),
            ScreeningRun(
                backbone=BERTIN,
                seed=13,
                dev_macro_f1=0.80,
                dev_accuracy=0.81,
                trainable_params=100,
                epoch_seconds=1.0,
                peak_vram_mib=10.0,
                cpu_latency_ms_p50=5.0,
                exportable=True,
                run_dir=Path("b"),
            ),
        ]


def test_screen_use_case_writes_screening_and_selection(tmp_path: Path) -> None:
    bundle = ScreenBackbones(_FakeRunner(), JsonReportWriter()).run(tmp_path)

    assert bundle.selection.selected == MRBERT
    screening = json.loads((tmp_path / "screening.json").read_text(encoding="utf-8"))
    selection = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    assert len(screening["runs"]) == 2
    assert selection["rule_branch"] == "mrbert_ahead"
