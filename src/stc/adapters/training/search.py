"""Optuna TPE search over the D7 LoRA space, tracked and resumable."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import torch

from stc.adapters.training.trainer import LoraTrainer
from stc.application.dto import SearchResult
from stc.application.ports import ExperimentTracker
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example

__all__ = ["OptunaSearcher", "SearchSpace"]

TrackerFactory = Callable[[Path], ExperimentTracker]


def _space_section(payload: Mapping[str, object]) -> Mapping[str, object]:
    section = payload.get("space", payload)
    if not isinstance(section, Mapping):
        raise InvalidInputError("the [space] section must be a table")
    return section


def _base_section(payload: Mapping[str, object]) -> Mapping[str, object]:
    section = payload.get("base", payload)
    if not isinstance(section, Mapping):
        raise InvalidInputError("the [base] section must be a table")
    return section


def _string_tuple(
    section: Mapping[str, object], key: str, default: Sequence[str]
) -> tuple[str, ...]:
    raw = section.get(key, list(default))
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise InvalidInputError(f"space.{key} must be a list of strings")
    return tuple(str(item) for item in raw)


def _int_tuple(section: Mapping[str, object], key: str, default: Sequence[int]) -> tuple[int, ...]:
    raw = section.get(key, list(default))
    if not isinstance(raw, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in raw
    ):
        raise InvalidInputError(f"space.{key} must be a list of integers")
    return tuple(int(item) for item in raw)


def _float_tuple(
    section: Mapping[str, object], key: str, default: Sequence[float]
) -> tuple[float, ...]:
    raw = section.get(key, list(default))
    if not isinstance(raw, list) or not all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in raw
    ):
        raise InvalidInputError(f"space.{key} must be a list of numbers")
    return tuple(float(item) for item in raw)


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """The categorical and bounded dimensions of D7."""

    r: tuple[int, ...] = (8, 16, 32, 64)
    alpha_multiplier: tuple[int, ...] = (1, 2)
    lora_dropout: tuple[float, ...] = (0.0, 0.05, 0.1)
    lr_min: float = 1e-4
    lr_max: float = 1e-3
    weight_decay: tuple[float, ...] = (0.0, 0.01, 0.1)
    epochs: tuple[int, ...] = (4, 6, 8, 10)
    batch_size: tuple[int, ...] = (16, 32, 64)
    warmup_ratio: tuple[float, ...] = (0.0, 0.06, 0.1)
    class_weighting: tuple[str, ...] = ("none", "sqrt_inv_freq")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> SearchSpace:
        """Validate a ``[space]`` TOML table, falling back to D7 defaults."""
        section = _space_section(payload)
        default = cls()
        lr = section.get("lr", [default.lr_min, default.lr_max])
        if (
            not isinstance(lr, list)
            or len(lr) != 2
            or not all(isinstance(item, int | float) and not isinstance(item, bool) for item in lr)
        ):
            raise InvalidInputError("space.lr must be a two-number [min, max] range")
        lr_min, lr_max = float(lr[0]), float(lr[1])
        if lr_min <= 0 or lr_min > lr_max:
            raise InvalidInputError("space.lr must satisfy 0 < min <= max")
        return cls(
            r=_int_tuple(section, "r", default.r),
            alpha_multiplier=_int_tuple(section, "alpha_multiplier", default.alpha_multiplier),
            lora_dropout=_float_tuple(section, "lora_dropout", default.lora_dropout),
            lr_min=lr_min,
            lr_max=lr_max,
            weight_decay=_float_tuple(section, "weight_decay", default.weight_decay),
            epochs=_int_tuple(section, "epochs", default.epochs),
            batch_size=_int_tuple(section, "batch_size", default.batch_size),
            warmup_ratio=_float_tuple(section, "warmup_ratio", default.warmup_ratio),
            class_weighting=_string_tuple(section, "class_weighting", default.class_weighting),
        )


def _toml_value(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return None


def write_best_config(path: Path, base: Mapping[str, object], params: Mapping[str, object]) -> None:
    """Write the best trial as a ``[base]`` TOML document usable by ``train lora``."""
    merged = {key: value for key, value in base.items() if key != "run_dir"}
    merged.update(params)
    lines = ["# generated by `stc train search` from the D7 space", "[base]"]
    for key in sorted(merged):
        rendered = _toml_value(merged[key])
        if rendered is not None:
            lines.append(f"{key} = {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class OptunaSearcher:
    """Run the D7 TPE search, optionally resuming its SQLite study."""

    def __init__(
        self,
        train: Sequence[Example],
        dev: Sequence[Example],
        out_dir: Path,
        base_config: Mapping[str, object],
        *,
        tracker_factory: TrackerFactory,
        device: torch.device | None = None,
        study_name: str = "stc-lora-search",
    ) -> None:
        self._train = list(train)
        self._dev = list(dev)
        self._out_dir = Path(out_dir)
        self._base = dict(_base_section(base_config))
        self._tracker_factory = tracker_factory
        self._device = device
        self._study_name = study_name

    def search(
        self,
        space: Mapping[str, object],
        n_trials: int,
        *,
        resume: bool = False,
    ) -> SearchResult:
        """Run at most ``n_trials`` total trials, returning the new trial count."""
        if n_trials < 1:
            raise InvalidInputError("n_trials must be positive")
        parsed = SearchSpace.from_mapping(space)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{self._out_dir / 'optuna.db'}"
        study = optuna.create_study(
            study_name=self._study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=13, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1),
            load_if_exists=True,
        )
        if not resume and study.trials:
            raise InvalidInputError(
                "an Optuna study already exists in this directory; pass --resume to continue"
            )
        before = len(study.trials)
        remaining = max(0, n_trials - before)
        if remaining:
            study.optimize(lambda trial: self._objective(trial, parsed), n_trials=remaining)
        after = len(study.trials)

        best_trial = _best_trial(study)
        best_params = self._materialise_params(dict(best_trial.params)) if best_trial else {}
        best_value = (
            float(best_trial.value) if best_trial and best_trial.value is not None else float("nan")
        )
        self._write_outputs(study, best_params, n_trials)
        return SearchResult(
            search_dir=self._out_dir,
            best_params=best_params,
            best_value=best_value,
            n_trials=after - before,
        )

    def _materialise_params(self, params: Mapping[str, Any]) -> dict[str, object]:
        if not params:
            return {}
        r = int(params["r"])
        multiplier = int(params.get("alpha_multiplier", 2))
        resolved = {key: value for key, value in params.items() if key != "alpha_multiplier"}
        resolved["r"] = r
        resolved["alpha"] = r * multiplier
        return resolved

    def _objective(self, trial: optuna.Trial, space: SearchSpace) -> float:
        multiplier = trial.suggest_categorical("alpha_multiplier", list(space.alpha_multiplier))
        r = trial.suggest_categorical("r", list(space.r))
        trial_params: dict[str, object] = {
            "r": r,
            "alpha": r * multiplier,
            "lora_dropout": trial.suggest_categorical("lora_dropout", list(space.lora_dropout)),
            "lr": trial.suggest_float("lr", space.lr_min, space.lr_max, log=True),
            "weight_decay": trial.suggest_categorical("weight_decay", list(space.weight_decay)),
            "epochs": trial.suggest_categorical("epochs", list(space.epochs)),
            "batch_size": trial.suggest_categorical("batch_size", list(space.batch_size)),
            "warmup_ratio": trial.suggest_categorical("warmup_ratio", list(space.warmup_ratio)),
            "class_weighting": trial.suggest_categorical(
                "class_weighting", list(space.class_weighting)
            ),
        }
        run_dir = self._out_dir / "trials" / f"trial-{trial.number:03d}"
        config: dict[str, object] = {
            **self._base,
            **trial_params,
            "method": "lora",
            "run_dir": str(run_dir),
            "save_artifacts": False,
        }

        def on_epoch_end(epoch: int, macro_f1: float) -> None:
            trial.report(macro_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned

        trainer = LoraTrainer(
            save_artifacts=False,
            on_epoch_end=on_epoch_end,
            device=self._device,
        )
        try:
            run = trainer.fit(
                self._train,
                self._dev,
                config,
                self._tracker_factory(run_dir),
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned from error
        return float(run.metrics["dev_macro_f1"])

    def _write_outputs(
        self,
        study: optuna.Study,
        best_params: Mapping[str, object],
        requested_trials: int,
    ) -> None:
        trials = [
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": dict(trial.params),
                "duration_seconds": (
                    float((trial.datetime_complete - trial.datetime_start).total_seconds())
                    if trial.datetime_complete is not None and trial.datetime_start is not None
                    else None
                ),
                "intermediate_values": {
                    str(step): float(value)
                    for step, value in sorted(trial.intermediate_values.items())
                },
            }
            for trial in study.trials
        ]
        best_trial = _best_trial(study)
        payload = {
            "schema_version": 1,
            "study_name": self._study_name,
            "direction": "maximize",
            "requested_trials": requested_trials,
            "n_trials": len(trials),
            "best": {
                "value": float(best_trial.value)
                if best_trial and best_trial.value is not None
                else None,
                "trial": best_trial.number if best_trial else None,
                "params": best_params,
            },
            "trials": trials,
        }
        (self._out_dir / "search_trials.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        if best_params:
            write_best_config(self._out_dir / "best_config.toml", self._base, best_params)
        (self._out_dir / "search_summary.md").write_text(
            _summary_markdown(payload), encoding="utf-8"
        )
        with contextlib.suppress(Exception):
            # Plotting is optional evidence; a figure failure must not fail the search.
            _write_figures(study, self._out_dir)


def _best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    try:
        return study.best_trial
    except ValueError:
        return None


def _summary_markdown(payload: Mapping[str, object]) -> str:
    trials = payload["trials"]
    if not isinstance(trials, list):
        return "# LoRA hyper-parameter search\n\nNo trials recorded.\n"
    lines = [
        "# LoRA hyper-parameter search",
        "",
        f"- Study: `{payload['study_name']}` (TPE, seed 13, multivariate)",
        f"- Trials: {payload['n_trials']} (requested {payload['requested_trials']})",
    ]
    best = payload["best"]
    if isinstance(best, Mapping) and best.get("value") is not None:
        lines.append(f"- Best dev macro-F1: {float(best['value']):.4f}")
        lines.append("")
        lines.append("## Best parameters")
        lines.append("")
        params = best.get("params")
        if isinstance(params, Mapping):
            for key in sorted(params):
                lines.append(f"- `{key}`: `{params[key]}`")
    lines.extend(["", "## Trials", "", "| trial | state | dev macro-F1 |", "|---|---|---|"])
    for trial in trials:
        if not isinstance(trial, Mapping):
            continue
        value = trial.get("value")
        shown = f"{float(value):.4f}" if isinstance(value, int | float) else "n/a"
        lines.append(f"| {trial.get('number')} | {trial.get('state')} | {shown} |")
    return "\n".join(lines) + "\n"


def _write_figures(study: optuna.Study, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [trial for trial in study.trials if trial.value is not None]
    if completed:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(
            [trial.number for trial in completed],
            [float(trial.value) for trial in completed],
            marker="o",
        )
        axis.set_xlabel("trial")
        axis.set_ylabel("dev macro-F1")
        axis.set_title("Optimisation history")
        figure.tight_layout()
        figure.savefig(out_dir / "search_history.png", dpi=120)
        plt.close(figure)
    if len(completed) >= 2:
        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception:
            importances = {}
        if importances:
            figure, axis = plt.subplots(figsize=(6, 4))
            names = list(importances)
            axis.barh(names, [importances[name] for name in names])
            axis.set_xlabel("importance")
            axis.set_title("Parameter importance")
            figure.tight_layout()
            figure.savefig(out_dir / "search_importance.png", dpi=120)
            plt.close(figure)
