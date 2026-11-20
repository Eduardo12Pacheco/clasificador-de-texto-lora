#!/usr/bin/env python
"""Validate the committed report tree against the plan's required shapes.

The checks are deliberately explicit (required keys, value types and ranges)
rather than a JSON-Schema dependency, so the script runs with the base install.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

VERDICTS = {"transformer_better", "baseline_better", "tie"}


class ReportError(ValueError):
    """A report file is missing or violates its required shape."""


def _load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ReportError(f"missing report: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"{path}: invalid JSON ({error})") from error
    if not isinstance(payload, dict):
        raise ReportError(f"{path}: expected a JSON object")
    return payload


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ReportError(f"{name}: expected an object")
    return value


def _list(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ReportError(f"{name}: expected a list")
    return value


def _number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReportError(f"{name}: expected a number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ReportError(f"{name}: {number} is below {minimum}")
    if maximum is not None and number > maximum:
        raise ReportError(f"{name}: {number} is above {maximum}")
    return number


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"{name}: expected an integer")
    if minimum is not None and value < minimum:
        raise ReportError(f"{name}: {value} is below {minimum}")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReportError(f"{name}: expected a non-empty string")
    return value


def _unit(value: object, name: str) -> float:
    return _number(value, name, minimum=0.0, maximum=1.0)


def _interval(value: object, name: str) -> None:
    interval = _mapping(value, name)
    estimate = _number(interval.get("estimate"), f"{name}.estimate")
    lower = _number(interval.get("lower"), f"{name}.lower")
    upper = _number(interval.get("upper"), f"{name}.upper")
    if not lower <= estimate <= upper:
        raise ReportError(f"{name}: estimate must lie in [lower, upper]")


def _metric_block(payload: Mapping[str, object], name: str) -> None:
    _integer(payload.get("n_examples"), f"{name}.n_examples", minimum=1)
    for key in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "top3_accuracy",
        "scenario_accuracy",
    ):
        _unit(payload.get(key), f"{name}.{key}")
    ci = _mapping(payload.get("ci"), f"{name}.ci")
    _interval(ci.get("accuracy"), f"{name}.ci.accuracy")
    _interval(ci.get("macro_f1"), f"{name}.ci.macro_f1")


def _validate_final_metrics(path: Path) -> None:
    payload = _load(path)
    _integer(payload.get("schema_version"), f"{path}.schema_version", minimum=1)
    if payload.get("split") != "test":
        raise ReportError(f"{path}: split must be 'test'")
    _integer(payload.get("n_examples"), f"{path}.n_examples", minimum=1)
    models = _mapping(payload.get("models"), f"{path}.models")
    for key in ("baseline", "lora"):
        _metric_block(_mapping(models.get(key), f"{path}.models.{key}"), f"{path}.{key}")


def _validate_calibration(path: Path) -> None:
    payload = _load(path)
    _integer(payload.get("schema_version"), f"{path}.schema_version", minimum=1)
    if payload.get("split") != "test":
        raise ReportError(f"{path}: split must be 'test'")
    models = _mapping(payload.get("models"), f"{path}.models")
    for key in ("baseline", "lora"):
        block = _mapping(models.get(key), f"{path}.models.{key}")
        _number(block.get("temperature"), f"{path}.{key}.temperature", minimum=0.0001)
        _unit(block.get("review_threshold"), f"{path}.{key}.review_threshold")
        before = _mapping(block.get("before"), f"{path}.{key}.before")
        _unit(before.get("ece"), f"{path}.{key}.before.ece")
        _number(before.get("nll"), f"{path}.{key}.before.nll", minimum=0.0)
        _number(before.get("brier"), f"{path}.{key}.before.brier", minimum=0.0)
        after = block.get("after")
        if after is not None:
            after_mapping = _mapping(after, f"{path}.{key}.after")
            _unit(after_mapping.get("ece"), f"{path}.{key}.after.ece")
            _number(after_mapping.get("nll"), f"{path}.{key}.after.nll", minimum=0.0)
            _number(after_mapping.get("brier"), f"{path}.{key}.after.brier", minimum=0.0)


def _validate_comparison(path: Path) -> None:
    payload = _load(path)
    _integer(payload.get("schema_version"), f"{path}.schema_version", minimum=1)
    if payload.get("split") != "test":
        raise ReportError(f"{path}: split must be 'test'")
    verdict = _string(payload.get("verdict"), f"{path}.verdict")
    if verdict not in VERDICTS:
        raise ReportError(f"{path}: unknown verdict {verdict!r}")
    delta = _mapping(payload.get("delta"), f"{path}.delta")
    _interval(delta.get("accuracy"), f"{path}.delta.accuracy")
    _interval(delta.get("macro_f1"), f"{path}.delta.macro_f1")
    mcnemar = _mapping(payload.get("mcnemar"), f"{path}.mcnemar")
    _integer(mcnemar.get("b"), f"{path}.mcnemar.b", minimum=0)
    _integer(mcnemar.get("c"), f"{path}.mcnemar.c", minimum=0)
    _unit(mcnemar.get("p"), f"{path}.mcnemar.p")
    sentences = _mapping(payload.get("verdict_sentences"), f"{path}.verdict_sentences")
    _string(sentences.get("en"), f"{path}.verdict_sentences.en")
    _string(sentences.get("es"), f"{path}.verdict_sentences.es")


def _validate_robustness(path: Path) -> None:
    payload = _load(path)
    _integer(payload.get("schema_version"), f"{path}.schema_version", minimum=1)
    models = _mapping(payload.get("models"), f"{path}.models")
    for key in ("baseline", "lora"):
        block = _mapping(models.get(key), f"{path}.models.{key}")
        rows = _list(block.get("perturbations"), f"{path}.{key}.perturbations")
        if not rows:
            raise ReportError(f"{path}.{key}.perturbations: must not be empty")
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"{path}.{key}.perturbations[{index}]")
            _string(row.get("name"), f"{path}.{key}.perturbations[{index}].name")
            _integer(row.get("n_affected"), f"{path}.{key}.perturbations[{index}].n_affected")
            _unit(row.get("accuracy_before"), f"{path}.{key}.perturbations[{index}].before")
            _unit(row.get("accuracy_after"), f"{path}.{key}.perturbations[{index}].after")
            lower = _number(row.get("delta_ci_lower"), f"{path}.{key}.delta_ci_lower")
            upper = _number(row.get("delta_ci_upper"), f"{path}.{key}.delta_ci_upper")
            if lower > upper:
                raise ReportError(f"{path}.{key}.perturbations[{index}]: CI is inverted")


def _validate_slices(path: Path) -> None:
    payload = _load(path)
    _integer(payload.get("schema_version"), f"{path}.schema_version", minimum=1)
    models = _mapping(payload.get("models"), f"{path}.models")
    for key in ("baseline", "lora"):
        block = _mapping(models.get(key), f"{path}.models.{key}")
        rows = _list(block.get("slices"), f"{path}.{key}.slices")
        if not rows:
            raise ReportError(f"{path}.{key}.slices: must not be empty")
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"{path}.{key}.slices[{index}]")
            _string(row.get("name"), f"{path}.{key}.slices[{index}].name")
            size = _integer(row.get("n"), f"{path}.{key}.slices[{index}].n", minimum=0)
            _unit(row.get("accuracy"), f"{path}.{key}.slices[{index}].accuracy")
            if row.get("macro_f1") is not None:
                _unit(row.get("macro_f1"), f"{path}.{key}.slices[{index}].macro_f1")
            if size >= 200 and row.get("macro_f1") is None:
                raise ReportError(f"{path}.{key}.slices[{index}]: macro_f1 is required at n >= 200")


def _validate_errors(path: Path) -> None:
    payload = _load(path)
    _integer(payload.get("schema_version"), f"{path}.schema_version", minimum=1)
    if payload.get("split") != "test":
        raise ReportError(f"{path}: split must be 'test'")
    rows = _list(payload.get("errors"), f"{path}.errors")
    _integer(payload.get("n_errors"), f"{path}.n_errors", minimum=0)
    if int(payload["n_errors"]) != len(rows):
        raise ReportError(f"{path}: n_errors does not match errors")
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{path}.errors[{index}]")
        _string(row.get("example_id"), f"{path}.errors[{index}].example_id")
        _string(row.get("text"), f"{path}.errors[{index}].text")
        _string(row.get("true_intent"), f"{path}.errors[{index}].true_intent")
        for model in ("baseline", "lora"):
            prediction = _mapping(row.get(model), f"{path}.errors[{index}].{model}")
            _string(
                prediction.get("predicted_intent"),
                f"{path}.errors[{index}].{model}.predicted_intent",
            )
            _unit(prediction.get("confidence"), f"{path}.errors[{index}].{model}.confidence")


def _validate_csv(path: Path, header: Sequence[str]) -> None:
    if not path.is_file():
        raise ReportError(f"missing report: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ReportError(f"{path}: empty CSV")
    if rows[0] != list(header):
        raise ReportError(f"{path}: header {rows[0]} != {list(header)}")
    if len(rows) < 2:
        raise ReportError(f"{path}: no data rows")


def validate_final(root: Path) -> None:
    """Validate every file required by the final report (S6-S10)."""
    directory = root / "final"
    _validate_final_metrics(directory / "metrics.json")
    _validate_comparison(directory / "comparison.json")
    _validate_calibration(directory / "calibration_test.json")
    _validate_robustness(directory / "robustness.json")
    _validate_slices(directory / "slices.json")
    _validate_errors(directory / "errors.json")
    _validate_csv(
        directory / "per_class.csv",
        ("model", "intent", "precision", "recall", "f1", "support"),
    )
    _validate_csv(
        directory / "confusion_matrix.csv",
        ("model", "true_intent", "predicted_intent", "count"),
    )


def validate_data_audit(root: Path) -> None:
    """Validate the leakage audit written by ``stc data audit``."""
    payload = _load(root / "data_audit.json")
    _integer(payload.get("schema_version"), "data_audit.schema_version", minimum=1)
    _string(payload.get("tarball_sha256"), "data_audit.tarball_sha256")
    for key in (
        "dev_test_overlap_texts",
        "dev_rows_in_train",
        "test_rows_in_train",
        "train_duplicate_groups",
        "train_after",
    ):
        _integer(payload.get(key), f"data_audit.{key}", minimum=0)
    _mapping(payload.get("processed_counts"), "data_audit.processed_counts")
    _mapping(payload.get("drops"), "data_audit.drops")


def validate_baseline(root: Path) -> None:
    """Validate the committed baseline grid and dev metrics."""
    directory = root / "baseline"
    search = _load(directory / "search.json")
    _integer(search.get("schema_version"), "baseline.search.schema_version", minimum=1)
    configs = _list(search.get("configs"), "baseline.search.configs")
    if not configs:
        raise ReportError("baseline.search.configs: must not be empty")
    for index, raw in enumerate(configs):
        config = _mapping(raw, f"baseline.search.configs[{index}]")
        _unit(config.get("dev_accuracy"), f"baseline.search.configs[{index}].dev_accuracy")
        _unit(config.get("dev_macro_f1"), f"baseline.search.configs[{index}].dev_macro_f1")
        _integer(config.get("n_features"), f"baseline.search.configs[{index}].n_features")
    _mapping(search.get("selection"), "baseline.search.selection")
    dev = _load(directory / "dev_metrics.json")
    _integer(dev.get("schema_version"), "baseline.dev_metrics.schema_version", minimum=1)
    for key in ("accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "scenario_accuracy"):
        _unit(dev.get(key), f"baseline.dev_metrics.{key}")


BACKBONE_BRANCHES = {"within_1pt_prefer_mrbert", "bertin_ahead", "mrbert_ahead"}


def _validate_environment(path: Path) -> None:
    payload = _load(path)
    _integer(payload.get("schema_version"), f"{path}.schema_version", minimum=1)
    _string(payload.get("git_sha"), f"{path}.git_sha")
    _string(payload.get("uv_lock_sha256"), f"{path}.uv_lock_sha256")
    _string(payload.get("device_name"), f"{path}.device_name")
    _string(payload.get("driver_version"), f"{path}.driver_version")
    if not isinstance(payload.get("cuda_available"), bool):
        raise ReportError(f"{path}.cuda_available: expected a boolean")
    versions = _mapping(payload.get("library_versions"), f"{path}.library_versions")
    for key in ("torch", "transformers", "peft", "optuna", "mlflow"):
        _string(versions.get(key), f"{path}.library_versions.{key}")
    if "stages_seconds" in payload:
        _mapping(payload.get("stages_seconds"), f"{path}.stages_seconds")


def _validate_screening(path: Path) -> None:
    payload = _load(path)
    _integer(payload.get("schema_version"), f"{path}.schema_version", minimum=1)
    runs = _list(payload.get("runs"), f"{path}.runs")
    if not runs:
        raise ReportError(f"{path}.runs: must not be empty")
    for index, raw in enumerate(runs):
        row = _mapping(raw, f"{path}.runs[{index}]")
        _string(row.get("backbone"), f"{path}.runs[{index}].backbone")
        _integer(row.get("seed"), f"{path}.runs[{index}].seed", minimum=0)
        _unit(row.get("dev_macro_f1"), f"{path}.runs[{index}].dev_macro_f1")
        _unit(row.get("dev_accuracy"), f"{path}.runs[{index}].dev_accuracy")
        _integer(row.get("trainable_params"), f"{path}.runs[{index}].trainable_params")
        _number(row.get("epoch_seconds"), f"{path}.runs[{index}].epoch_seconds", minimum=0.0)
        _number(row.get("peak_vram_mib"), f"{path}.runs[{index}].peak_vram_mib", minimum=0.0)
        _number(
            row.get("cpu_latency_ms_p50"),
            f"{path}.runs[{index}].cpu_latency_ms_p50",
            minimum=0.0,
        )


def _validate_selection(path: Path) -> None:
    payload = _load(path)
    _string(payload.get("selected"), f"{path}.selected")
    branch = _string(payload.get("rule_branch"), f"{path}.rule_branch")
    if branch not in BACKBONE_BRANCHES:
        raise ReportError(f"{path}.rule_branch: unknown branch {branch!r}")
    _mapping(payload.get("per_backbone_means"), f"{path}.per_backbone_means")


def _validate_search_trials(path: Path) -> None:
    payload = _load(path)
    _integer(payload.get("schema_version"), f"{path}.schema_version", minimum=1)
    trials = _list(payload.get("trials"), f"{path}.trials")
    if not trials:
        raise ReportError(f"{path}.trials: must not be empty")
    for index, raw in enumerate(trials):
        trial = _mapping(raw, f"{path}.trials[{index}]")
        _integer(trial.get("number"), f"{path}.trials[{index}].number", minimum=0)
        _string(trial.get("state"), f"{path}.trials[{index}].state")
        _mapping(trial.get("params"), f"{path}.trials[{index}].params")
    best = _mapping(payload.get("best"), f"{path}.best")
    _number(best.get("value"), f"{path}.best.value", minimum=0.0, maximum=1.0)


def _validate_full_ft(path: Path) -> None:
    payload = _load(path)
    _integer(payload.get("schema_version"), f"{path}.schema_version", minimum=1)
    runs = _list(payload.get("runs"), f"{path}.runs")
    if not runs:
        raise ReportError(f"{path}.runs: must not be empty")
    for index, raw in enumerate(runs):
        row = _mapping(raw, f"{path}.runs[{index}]")
        _number(row.get("lr"), f"{path}.runs[{index}].lr", minimum=0.0)
        _unit(row.get("dev_macro_f1"), f"{path}.runs[{index}].dev_macro_f1")


def _validate_final_seeds(path: Path) -> None:
    payload = _load(path)
    _integer(payload.get("schema_version"), f"{path}.schema_version", minimum=1)
    seeds = _list(payload.get("seeds"), f"{path}.seeds")
    if not seeds:
        raise ReportError(f"{path}.seeds: must not be empty")
    _integer(payload.get("selected_seed"), f"{path}.selected_seed", minimum=0)
    _unit(payload.get("dev_macro_f1_mean"), f"{path}.dev_macro_f1_mean")
    _number(payload.get("dev_macro_f1_std"), f"{path}.dev_macro_f1_std", minimum=0.0)


def validate_experiments(root: Path) -> None:
    """Validate the compact experiment reports committed by T11."""
    _validate_environment(root / "environment.json")
    _validate_screening(root / "screening.json")
    _validate_selection(root / "selection.json")
    _validate_full_ft(root / "full_ft_reference.json")
    _validate_search_trials(root / "search_trials.json")
    _validate_final_seeds(root / "final_seeds.json")


def validate_all(root: Path) -> None:
    """Validate every report family that exists in the committed tree."""
    validate_data_audit(root)
    validate_baseline(root)
    validate_final(root)


_VALIDATORS: dict[str, Callable[[Path], None]] = {
    "final": validate_final,
    "experiments": validate_experiments,
    "all": validate_all,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested validator; return 0 on success and 1 on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(_VALIDATORS))
    parser.add_argument("--root", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    try:
        _VALIDATORS[args.kind](args.root)
    except ReportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"OK {args.kind} reports under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
