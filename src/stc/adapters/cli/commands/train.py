"""``stc train``: classical baseline, LoRA seeds, full-FT, search and screening."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, cast

import typer

from stc import bootstrap
from stc.adapters.cli.errors import handle_domain_errors
from stc.application.ports import ExperimentTracker
from stc.application.services.dev_only_repository import DevOnlyRepository
from stc.application.use_cases.screen_backbones import ScreenBackbones
from stc.application.use_cases.search_hyperparameters import SearchHyperparameters
from stc.application.use_cases.train_baseline import load_baseline_grid
from stc.application.use_cases.train_transformer import (
    TrainTransformer,
    load_toml_config,
)
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.splits import Split

_DATASETS = {"massive", "fixture"}

train_app = typer.Typer(
    name="train",
    help="Train the classical baseline or a transformer classifier.",
    no_args_is_help=True,
)


def _format_params(params: dict[str, object]) -> str:
    parts = [f"classifier={params.get('classifier')}"]
    for key in ("word_ngram", "char_ngram", "C", "class_weight"):
        if key in params:
            parts.append(f"{key}={params[key]}")
    return " ".join(parts)


def _parse_ints(value: str, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise InvalidInputError(f"{name} must be a comma-separated list of integers") from error
    if not parsed:
        raise InvalidInputError(f"{name} must not be empty")
    return parsed


def _parse_floats(value: str, name: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise InvalidInputError(f"{name} must be a comma-separated list of numbers") from error
    if not parsed:
        raise InvalidInputError(f"{name} must not be empty")
    return parsed


def _base_section(payload: Mapping[str, object]) -> dict[str, object]:
    section = payload.get("base", payload)
    if not isinstance(section, Mapping):
        raise InvalidInputError("the [base] section must be a table")
    return {str(key): value for key, value in section.items()}


def _dataset(value: str) -> str:
    if value not in _DATASETS:
        raise InvalidInputError(f"unknown dataset: {value!r}; expected massive or fixture")
    return value


def _splits(dataset: str) -> tuple[list[Example], list[Example]]:
    repository = DevOnlyRepository(bootstrap.build_training_repository(dataset=dataset))
    train = list(repository.load(Split.TRAIN))
    dev = list(repository.load(Split.DEV))
    if not train or not dev:
        raise InvalidInputError("training needs non-empty train and dev splits")
    return train, dev


def _tracker_factory(tracker: str | None) -> Callable[[Path], ExperimentTracker]:
    settings = bootstrap.build_settings()
    factory = bootstrap.build_tracker_factory(settings)
    spec = tracker or settings.tracker
    return lambda run_dir: factory.create(spec, run_dir)


def _identity() -> tuple[str, str]:
    return bootstrap.repository_identity()


@train_app.command("baseline")
@handle_domain_errors
def train_baseline(
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset to read: massive (default) or fixture."),
    ] = "massive",
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for the fitted baseline artifacts."),
    ] = Path("artifacts/baseline"),
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="TOML grid file (defaults to configs/baseline.toml when present).",
        ),
    ] = None,
    quick: Annotated[
        bool,
        typer.Option("--quick", help="Fit only the first two SVC configs (smoke tests)."),
    ] = False,
    reports: Annotated[
        Path | None,
        typer.Option(
            "--reports",
            help="Directory for search.json and dev_metrics.json (defaults to --out).",
        ),
    ] = None,
) -> None:
    """Grid-tune the TF-IDF + linear baseline on dev and persist the winner."""
    grid = load_baseline_grid(config)
    services = bootstrap.build_baseline_services(dataset=_dataset(dataset))
    result = services.use_case.run(
        services.repository,
        out_dir=out,
        grid=grid,
        quick=quick,
        reports_dir=reports,
    )
    winner = dict(result.winner.params)
    macro_f1 = float(result.dev_metrics["macro_f1"])  # type: ignore[arg-type]
    accuracy = float(result.dev_metrics["accuracy"])  # type: ignore[arg-type]
    typer.echo(f"winner: {_format_params(winner)}")
    typer.echo(f"dev macro_f1={macro_f1:.4f} accuracy={accuracy:.4f} n_configs={result.n_configs}")
    typer.echo(f"wrote {result.artifact_dir} and {result.search_path}")


@train_app.command("lora")
@handle_domain_errors
def train_lora(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML training recipe (must contain a [base] table)."),
    ] = Path("configs/smoke.toml"),
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset to read: massive (default) or fixture."),
    ] = "massive",
    backbone: Annotated[
        str | None,
        typer.Option("--backbone", help="Override the backbone id from the config."),
    ] = None,
    seeds: Annotated[
        str,
        typer.Option("--seeds", help="Comma-separated seeds, e.g. 13,42,2026."),
    ] = "13",
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for seed runs and selected.json."),
    ] = Path("artifacts/final"),
    tracker: Annotated[
        str | None,
        typer.Option("--tracker", help="Comma-separated trackers: jsonl,mlflow,null."),
    ] = None,
) -> None:
    """Fine-tune LoRA on every seed, select the shipped seed on dev and persist it."""
    bootstrap.require_train_extra()
    parsed = load_toml_config(config)
    base = _base_section(parsed)
    if backbone is not None:
        base["backbone"] = backbone
    git_sha, lock_sha = _identity()
    base["git_sha"] = git_sha
    base["uv_lock_sha256"] = lock_sha
    trainer = bootstrap.build_trainer("lora")
    use_case = TrainTransformer(trainer, bootstrap.build_report_writer())
    result = use_case.run_lora(
        bootstrap.build_training_repository(dataset=_dataset(dataset)),
        {"base": base},
        seeds=_parse_ints(seeds, "--seeds"),
        out_dir=out,
        tracker_factory=_tracker_factory(tracker),
        dataset=dataset,
    )
    for row in result.seeds:
        typer.echo(f"seed={row['seed']} dev macro_f1={float(cast(float, row['dev_macro_f1'])):.4f}")
    typer.echo(f"dev macro_f1={result.mean_macro_f1:.4f} selected_seed={result.selected_seed}")
    typer.echo(f"wrote {result.out_dir / 'selected.json'}")


@train_app.command("full-ft")
@handle_domain_errors
def train_full_ft(
    backbone: Annotated[
        str,
        typer.Option("--backbone", help="Hugging Face backbone id."),
    ],
    lrs: Annotated[
        str,
        typer.Option("--lrs", help="Comma-separated learning rates, e.g. 2e-5,3e-5,5e-5."),
    ] = "2e-5,3e-5,5e-5",
    epochs: Annotated[
        int,
        typer.Option("--epochs", help="Epochs per learning rate (D7 fixes 4)."),
    ] = 4,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset to read: massive (default) or fixture."),
    ] = "massive",
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for per-lr runs and full_ft.json."),
    ] = Path("artifacts/full_ft"),
    tracker: Annotated[
        str | None,
        typer.Option("--tracker", help="Comma-separated trackers: jsonl,mlflow,null."),
    ] = None,
) -> None:
    """Run the full fine-tuning reference grid on dev (D7)."""
    bootstrap.require_train_extra()
    git_sha, lock_sha = _identity()
    base: dict[str, object] = {
        "backbone": backbone,
        "max_length": 64,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "grad_accum": 1,
        "class_weighting": "none",
        "git_sha": git_sha,
        "uv_lock_sha256": lock_sha,
    }
    trainer = bootstrap.build_trainer("full")
    use_case = TrainTransformer(trainer, bootstrap.build_report_writer())
    result = use_case.run_full_ft(
        bootstrap.build_training_repository(dataset=_dataset(dataset)),
        {"base": base},
        backbone=backbone,
        learning_rates=_parse_floats(lrs, "--lrs"),
        epochs=epochs,
        out_dir=out,
        tracker_factory=_tracker_factory(tracker),
        dataset=dataset,
    )
    typer.echo(f"best_lr={result.best_lr:g} dev macro_f1={result.best_macro_f1:.4f}")
    typer.echo(f"wrote {result.out_dir / 'full_ft.json'}")


@train_app.command("search")
@handle_domain_errors
def train_search(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML file with [base], [space] and [search]."),
    ] = Path("configs/lora_search.toml"),
    n_trials: Annotated[
        int | None,
        typer.Option("--trials", help="Total trial budget (defaults to the config value)."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Continue an existing Optuna study."),
    ] = False,
    backbone: Annotated[
        str | None,
        typer.Option("--backbone", help="Override the backbone id from the config."),
    ] = None,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset to read: massive (default) or fixture."),
    ] = "massive",
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for the Optuna study and search reports."),
    ] = Path("artifacts/search"),
    tracker: Annotated[
        str | None,
        typer.Option("--tracker", help="Comma-separated trackers: jsonl,mlflow,null."),
    ] = None,
) -> None:
    """Search the D7 LoRA space with Optuna TPE and write the search reports."""
    bootstrap.require_train_extra()
    parsed = dict(load_toml_config(config))
    base = _base_section(parsed)
    if backbone is not None:
        base["backbone"] = backbone
    git_sha, lock_sha = _identity()
    base["git_sha"] = git_sha
    base["uv_lock_sha256"] = lock_sha
    search_section = parsed.get("search", {})
    configured_trials = 24
    if isinstance(search_section, Mapping) and "n_trials" in search_section:
        configured_trials = int(search_section["n_trials"])
    trials = n_trials if n_trials is not None else configured_trials
    train, dev = _splits(_dataset(dataset))
    searcher = bootstrap.build_searcher(
        train,
        dev,
        out,
        {"base": base},
        tracker_factory=_tracker_factory(tracker),
    )
    run = SearchHyperparameters(searcher).run(parsed, n_trials=trials, resume=resume)
    typer.echo(f"best dev macro_f1={run.result.best_value:.4f}")
    typer.echo(f"{run.new_trials} new trials")
    typer.echo(f"wrote {run.out_dir / 'search_trials.json'}")


@train_app.command("screen")
@handle_domain_errors
def train_screen(
    config: Annotated[
        Path,
        typer.Option("--config", help="TOML screening recipe with [base] and [screening]."),
    ] = Path("configs/screening.toml"),
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset to read: massive (default) or fixture."),
    ] = "massive",
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for per-run artifacts and selection.json."),
    ] = Path("artifacts/screening"),
    tracker: Annotated[
        str | None,
        typer.Option("--tracker", help="Comma-separated trackers: jsonl,mlflow,null."),
    ] = None,
) -> None:
    """Screen the candidate backbones x seeds and apply the D6 selection rule."""
    bootstrap.require_train_extra()
    parsed = dict(load_toml_config(config))
    train, dev = _splits(_dataset(dataset))
    runner = bootstrap.build_screener(
        train,
        dev,
        out,
        parsed,
        tracker_factory=_tracker_factory(tracker),
    )
    bundle = ScreenBackbones(runner, bootstrap.build_report_writer()).run(out)
    typer.echo(f"selected={bundle.selection.selected} rule_branch={bundle.selection.rule_branch}")
    typer.echo(f"wrote {bundle.out_dir / 'screening.json'}")
    typer.echo("SCREENING DONE")
