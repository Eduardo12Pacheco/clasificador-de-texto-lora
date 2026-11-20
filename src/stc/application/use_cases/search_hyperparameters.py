"""Thin use case around the Optuna search port (D7)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from stc.application.dto import SearchResult
from stc.application.ports import HyperparameterSearcher

__all__ = ["SearchHyperparameters", "SearchRun"]


@dataclass(frozen=True, slots=True)
class SearchRun:
    """Output paths and the new-trial count of one search invocation."""

    result: SearchResult
    out_dir: Path

    @property
    def new_trials(self) -> int:
        """How many trials this invocation actually added."""
        return self.result.n_trials


class SearchHyperparameters:
    """Run or resume a search from the CLI-independent layers."""

    def __init__(self, searcher: HyperparameterSearcher) -> None:
        self._searcher = searcher

    def run(
        self,
        config: Mapping[str, object],
        *,
        n_trials: int,
        resume: bool = False,
    ) -> SearchRun:
        """Delegate to the configured search adapter."""
        result = self._searcher.search(config, n_trials, resume=resume)
        return SearchRun(result=result, out_dir=result.search_dir)
