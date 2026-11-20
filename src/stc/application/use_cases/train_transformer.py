"""Training use cases: LoRA seeds and the full fine-tuning reference (D7)."""

from __future__ import annotations

import json
import statistics
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from stc.application.ports import (
    ClassifierTrainer,
    DatasetRepository,
    ExperimentTracker,
    JsonSink,
)
from stc.application.services.dev_only_repository import DevOnlyRepository
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.splits import Split

__all__ = [
    "FullFtResult",
    "LoraSeedsResult",
    "TrainTransformer",
    "load_toml_config",
]

TrackerFactory = Callable[[Path], ExperimentTracker]

SELECTION_RULE = "shipped seed = max dev macro-F1; tie broken by lower dev NLL (D7)"


def load_toml_config(path: Path) -> Mapping[str, object]:
    """Read a training TOML file as a mapping."""
    target = Path(path)
    if not target.is_file():
        raise InvalidInputError(f"training config not found: {target}")
    try:
        with target.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise InvalidInputError(f"cannot read training config {target}: {error}") from error
    return payload


def _base(payload: Mapping[str, object]) -> dict[str, object]:
    section = payload.get("base", payload)
    if not isinstance(section, Mapping):
        raise InvalidInputError("the [base] section must be a table")
    return dict(section)


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidInputError(f"cannot read run artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise InvalidInputError(f"run artifact {path} is not a JSON object")
    return payload


@dataclass(frozen=True, slots=True)
class LoraSeedsResult:
    """Outcome of training one LoRA configuration across seeds."""

    selected_seed: int
    selected_run: Path
    seeds: tuple[Mapping[str, object], ...]
    mean_macro_f1: float
    std_macro_f1: float
    out_dir: Path


@dataclass(frozen=True, slots=True)
class FullFtResult:
    """Outcome of the full fine-tuning reference grid."""

    best_lr: float
    best_macro_f1: float
    runs: tuple[Mapping[str, object], ...]
    out_dir: Path


def _fill_run_config(
    base: Mapping[str, object],
    *,
    run_dir: Path,
    seed: int,
    dataset: str,
    method: str,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    config = dict(base)
    config.update(
        {
            "run_dir": str(run_dir),
            "seed": seed,
            "dataset": dataset,
            "method": method,
        }
    )
    if overrides:
        config.update(overrides)
    return config


def _run_metrics(run_dir: Path) -> Mapping[str, object]:
    return _read_json(run_dir / "metrics.json")


def _number_field(payload: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)


class TrainTransformer:
    """Train LoRA across seeds or the full-FT reference, skipping finished runs."""

    def __init__(self, trainer: ClassifierTrainer, writer: JsonSink) -> None:
        self._trainer = trainer
        self._writer = writer

    def run_lora(
        self,
        repository: DatasetRepository,
        config: Mapping[str, object],
        *,
        seeds: Sequence[int],
        out_dir: Path,
        tracker_factory: TrackerFactory,
        dataset: str = "unknown",
    ) -> LoraSeedsResult:
        """Train every seed under ``<out>/seed-<n>/`` and select the shipped seed."""
        if not seeds:
            raise InvalidInputError("at least one seed is required")
        train, dev = self._splits(repository)
        base = _base(config)
        rows: list[dict[str, object]] = []
        for seed in seeds:
            run_dir = Path(out_dir) / f"seed-{seed}"
            if not (run_dir / "run.json").is_file():
                payload = _fill_run_config(
                    base,
                    run_dir=run_dir,
                    seed=seed,
                    dataset=dataset,
                    method="lora",
                )
                self._trainer.fit(
                    train,
                    dev,
                    payload,
                    tracker_factory(run_dir),
                )
            metrics = _run_metrics(run_dir)
            rows.append(
                {
                    "seed": seed,
                    "run": str(run_dir),
                    "dev_macro_f1": _number_field(metrics, "dev_macro_f1"),
                    "dev_nll": _number_field(metrics, "dev_nll"),
                    "dev_accuracy": _number_field(metrics, "dev_accuracy"),
                }
            )
        selected = max(
            rows,
            key=lambda row: (_number_field(row, "dev_macro_f1"), -_number_field(row, "dev_nll")),
        )
        values = [_number_field(row, "dev_macro_f1") for row in rows]
        mean = float(statistics.fmean(values))
        std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        selected_payload = {
            "schema_version": 1,
            "rule": SELECTION_RULE,
            "selected_seed": selected["seed"],
            "selected_run": selected["run"],
            "seeds": rows,
            "dev_macro_f1_mean": mean,
            "dev_macro_f1_std": std,
        }
        final_payload = {
            "schema_version": 1,
            "seeds": rows,
            "selected_seed": selected["seed"],
            "selected_run": selected["run"],
            "dev_macro_f1_mean": mean,
            "dev_macro_f1_std": std,
        }
        self._writer.write_json(destination / "selected.json", selected_payload)
        self._writer.write_json(destination / "final_seeds.json", final_payload)
        return LoraSeedsResult(
            selected_seed=int(str(selected["seed"])),
            selected_run=Path(str(selected["run"])),
            seeds=tuple(rows),
            mean_macro_f1=mean,
            std_macro_f1=std,
            out_dir=destination,
        )

    def run_full_ft(
        self,
        repository: DatasetRepository,
        config: Mapping[str, object],
        *,
        backbone: str,
        learning_rates: Sequence[float],
        epochs: int,
        out_dir: Path,
        tracker_factory: TrackerFactory,
        dataset: str = "unknown",
    ) -> FullFtResult:
        """Run one full fine-tuning reference per learning rate (seed 13)."""
        if not learning_rates:
            raise InvalidInputError("at least one learning rate is required")
        train, dev = self._splits(repository)
        base = _base(config)
        base.setdefault("backbone", backbone)
        rows: list[dict[str, object]] = []
        for lr in learning_rates:
            run_dir = Path(out_dir) / f"lr-{lr:g}"
            if not (run_dir / "run.json").is_file():
                payload = _fill_run_config(
                    base,
                    run_dir=run_dir,
                    seed=13,
                    dataset=dataset,
                    method="full",
                    overrides={
                        "backbone": backbone,
                        "lr": lr,
                        "epochs": epochs,
                        "batch_size": base.get("batch_size", 32),
                    },
                )
                self._trainer.fit(
                    train,
                    dev,
                    payload,
                    tracker_factory(run_dir),
                )
            metrics = _run_metrics(run_dir)
            rows.append(
                {
                    "lr": lr,
                    "run": str(run_dir),
                    "dev_macro_f1": _number_field(metrics, "dev_macro_f1"),
                    "dev_nll": _number_field(metrics, "dev_nll"),
                    "dev_accuracy": _number_field(metrics, "dev_accuracy"),
                }
            )
        selected = max(rows, key=lambda row: _number_field(row, "dev_macro_f1"))
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self._writer.write_json(
            destination / "full_ft.json",
            {
                "schema_version": 1,
                "backbone": backbone,
                "epochs": epochs,
                "runs": rows,
                "best": {
                    "lr": selected["lr"],
                    "dev_macro_f1": selected["dev_macro_f1"],
                },
            },
        )
        return FullFtResult(
            best_lr=_number_field(selected, "lr"),
            best_macro_f1=_number_field(selected, "dev_macro_f1"),
            runs=tuple(rows),
            out_dir=destination,
        )

    @staticmethod
    def _splits(repository: DatasetRepository) -> tuple[list[Example], list[Example]]:
        guarded = DevOnlyRepository(repository)
        train = list(guarded.load(Split.TRAIN))
        dev = list(guarded.load(Split.DEV))
        if not train or not dev:
            raise InvalidInputError("training needs non-empty train and dev splits")
        return train, dev
