"""End-to-end baseline training on the bundled fixture through the CLI."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stc.adapters.cli.app import app

pytestmark = pytest.mark.integration

runner = CliRunner()


def _normalise_search(payload: dict[str, object]) -> dict[str, object]:
    for candidate in payload["configs"]:  # type: ignore[union-attr]
        candidate["fit_seconds"] = 0.0
    return payload


def test_train_baseline_quick_on_the_fixture(tmp_path: Path) -> None:
    out = tmp_path / "baseline"
    started = time.perf_counter()

    result = runner.invoke(
        app,
        ["train", "baseline", "--dataset", "fixture", "--quick", "--out", str(out)],
    )
    elapsed = time.perf_counter() - started

    assert result.exit_code == 0, result.output
    assert "winner" in result.output
    assert "dev macro_f1" in result.output
    assert elapsed < 20
    for name in ("baseline.skops", "labels.json", "manifest.json", "search.json"):
        assert (out / name).is_file()
    search = json.loads((out / "search.json").read_text(encoding="utf-8"))
    assert search["quick"] is True
    assert search["n_configs"] == 2
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "tfidf-linear"
    assert manifest["quick"] is True


def test_evaluate_dev_matches_the_committed_dev_metrics(tmp_path: Path) -> None:
    out = tmp_path / "baseline"
    train = runner.invoke(
        app,
        ["train", "baseline", "--dataset", "fixture", "--quick", "--out", str(out)],
    )
    assert train.exit_code == 0, train.output

    evaluated = runner.invoke(
        app,
        ["evaluate", "--split", "dev", "--dataset", "fixture", "--model", str(out), "--json"],
    )

    assert evaluated.exit_code == 0, evaluated.output
    payload = json.loads(evaluated.output)
    dev_metrics = json.loads((out / "dev_metrics.json").read_text(encoding="utf-8"))
    assert payload["macro_f1"] == pytest.approx(dev_metrics["macro_f1"], abs=1e-9)
    assert payload["accuracy"] == pytest.approx(dev_metrics["accuracy"], abs=1e-9)


def test_two_quick_runs_are_deterministic_apart_from_fit_seconds(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for out in (first, second):
        result = runner.invoke(
            app,
            ["train", "baseline", "--dataset", "fixture", "--quick", "--out", str(out)],
        )
        assert result.exit_code == 0, result.output

    search_one = _normalise_search(json.loads((first / "search.json").read_text(encoding="utf-8")))
    search_two = _normalise_search(json.loads((second / "search.json").read_text(encoding="utf-8")))
    assert search_one == search_two
    assert (first / "dev_metrics.json").read_text(encoding="utf-8") == (
        second / "dev_metrics.json"
    ).read_text(encoding="utf-8")


def test_unknown_dataset_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["train", "baseline", "--dataset", "nope", "--quick", "--out", str(tmp_path)],
    )

    assert result.exit_code == 2
