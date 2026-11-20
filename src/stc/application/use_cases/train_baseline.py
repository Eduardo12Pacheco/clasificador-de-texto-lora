"""Train the D5 TF-IDF + linear baseline grid and persist the winner."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from stc.application.dto import (
    BaselineCandidate,
    BaselineGrid,
    DatasetMeta,
)
from stc.application.ports import (
    BaselineStore,
    BaselineTrainer,
    DatasetRepository,
    JsonSink,
)
from stc.application.services.baseline_selection import SELECTION_RULE, select_baseline_winner
from stc.application.services.dev_only_repository import DevOnlyRepository
from stc.application.use_cases.evaluate_model import EvaluateModel
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace
from stc.domain.splits import Split

DEFAULT_GRID_PATH = Path("configs/baseline.toml")

__all__ = [
    "DEFAULT_GRID_PATH",
    "TrainBaseline",
    "TrainBaselineResult",
    "load_baseline_grid",
]


@dataclass(frozen=True, slots=True)
class TrainBaselineResult:
    """The selected configuration and every artifact the run wrote."""

    winner: BaselineCandidate
    dev_metrics: Mapping[str, object]
    artifact_dir: Path
    search_path: Path
    dev_metrics_path: Path
    manifest: Mapping[str, object]
    n_configs: int


def load_baseline_grid(path: Path | None) -> BaselineGrid:
    """Load the grid from ``path`` or from ``configs/baseline.toml`` when present."""
    target = Path(path) if path is not None else DEFAULT_GRID_PATH
    if path is None and not target.is_file():
        return BaselineGrid.default()
    if not target.is_file():
        raise InvalidInputError(f"baseline config not found: {target}")
    try:
        with target.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise InvalidInputError(f"cannot read baseline config {target}: {error}") from error
    return BaselineGrid.from_toml_table(payload)


def _label_space(train: Sequence[Example], dev: Sequence[Example]) -> LabelSpace:
    return LabelSpace.from_intents(sorted({row.intent for row in (*train, *dev)}))


def _search_payload(
    candidates: Sequence[BaselineCandidate],
    grid: BaselineGrid,
    winner: BaselineCandidate,
    meta: DatasetMeta,
    *,
    quick: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": meta.to_dict(),
        "grid": grid.to_dict(),
        "quick": quick,
        "n_configs": len(candidates),
        "configs": [candidate.to_dict() for candidate in candidates],
        "selection": {
            "rule": SELECTION_RULE,
            "winner": dict(winner.params),
            "dev_macro_f1": winner.dev_macro_f1,
            "dev_accuracy": winner.dev_accuracy,
            "n_features": winner.n_features,
        },
    }


class TrainBaseline:
    """Fit the grid on train, select on dev and persist the winner."""

    def __init__(
        self,
        trainer: BaselineTrainer,
        store: BaselineStore,
        writer: JsonSink,
    ) -> None:
        self._trainer = trainer
        self._store = store
        self._writer = writer

    def run(
        self,
        repository: DatasetRepository,
        *,
        out_dir: Path,
        grid: BaselineGrid,
        quick: bool = False,
        reports_dir: Path | None = None,
    ) -> TrainBaselineResult:
        """Run the grid; dev and test stay behind their access policies."""
        guarded = DevOnlyRepository(repository)
        train = list(guarded.load(Split.TRAIN))
        dev = list(guarded.load(Split.DEV))
        if not train:
            raise InvalidInputError("baseline training needs a non-empty train split")
        if not dev:
            raise InvalidInputError("baseline training needs a non-empty dev split")
        meta = guarded.meta()
        labels = _label_space(train, dev)

        candidates = tuple(
            self._trainer.evaluate_configs(
                train,
                dev,
                labels,
                grid,
                quick=quick,
            )
        )
        winner = select_baseline_winner(candidates)
        classifier = self._trainer.fit(train, labels, winner.params)
        evaluation = EvaluateModel().run(classifier, dev, split=Split.DEV)

        artifact_dir = Path(out_dir)
        destination = Path(reports_dir) if reports_dir is not None else artifact_dir
        destination.mkdir(parents=True, exist_ok=True)
        search_path = destination / "search.json"
        dev_metrics_path = destination / "dev_metrics.json"

        manifest = self._store.save(
            classifier,
            artifact_dir,
            metadata={
                "grid": grid.to_dict(),
                "winner": dict(winner.params),
                "dev_accuracy": winner.dev_accuracy,
                "dev_macro_f1": winner.dev_macro_f1,
                "n_features": winner.n_features,
                "dataset": meta.to_dict(),
                "quick": quick,
                "n_configs": len(candidates),
            },
        )

        dev_metrics = evaluation.to_dict()
        dev_metrics["winner"] = dict(winner.params)
        dev_metrics["n_features"] = winner.n_features
        dev_metrics["grid"] = {"quick": quick, "n_configs": len(candidates)}
        self._writer.write_json(
            search_path,
            _search_payload(candidates, grid, winner, meta, quick=quick),
        )
        self._writer.write_json(dev_metrics_path, dev_metrics)

        return TrainBaselineResult(
            winner=winner,
            dev_metrics=dev_metrics,
            artifact_dir=artifact_dir,
            search_path=search_path,
            dev_metrics_path=dev_metrics_path,
            manifest=manifest,
            n_configs=len(candidates),
        )
