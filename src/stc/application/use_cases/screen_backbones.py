"""Screen backbones and persist the D6 selection decision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from stc.application.dto import ScreeningRun
from stc.application.ports import JsonSink, ScreeningRunner
from stc.application.services.backbone_selection import BackboneSelection, select_backbone

__all__ = ["ScreenBackbones", "ScreeningBundle"]


@dataclass(frozen=True, slots=True)
class ScreeningBundle:
    """All screening rows and the resulting selection."""

    runs: Sequence[ScreeningRun]
    selection: BackboneSelection
    out_dir: Path


class ScreenBackbones:
    """Run screening, then write ``screening.json`` and ``selection.json``."""

    def __init__(self, runner: ScreeningRunner, writer: JsonSink) -> None:
        self._runner = runner
        self._writer = writer

    def run(self, out_dir: Path) -> ScreeningBundle:
        """Execute the runner and persist its contract artifacts."""
        runs = list(self._runner.run())
        selection = select_backbone(runs)
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self._writer.write_json(
            destination / "screening.json",
            {
                "schema_version": 1,
                "runs": [run.to_dict() for run in runs],
            },
        )
        self._writer.write_json(destination / "selection.json", selection.to_dict())
        return ScreeningBundle(runs=runs, selection=selection, out_dir=destination)
