"""CLI tests for ``stc evaluate`` and its test-split lock."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from stc.adapters.cli.app import app
from stc.adapters.cli.commands import evaluate as evaluate_module
from stc.application.dto import ModelInfo
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace

pytestmark = pytest.mark.unit

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class FakeClassifier:
    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            model_id="fake-model",
            kind="fake",
            version="0.0.1",
            quantization=None,
            parameters=10,
            license="MIT",
            n_intents=3,
            n_scenarios=2,
        )

    @property
    def labels(self) -> LabelSpace:
        return LabelSpace.from_intents(("alarm_set", "alarm_cancel", "play_music"))

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        # Sorted label space: alarm_cancel, alarm_set, play_music.
        return np.array([[0.0, 2.0, 0.0] for _ in texts], dtype=np.float32)


def make_examples() -> list[Example]:
    return [
        Example("1", "pon una alarma", "alarm_set", "alarm", "w1"),
        Example("2", "pon música", "play_music", "play", "w1"),
    ]


def test_test_split_without_final_exits_six_before_model_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_: str) -> FakeClassifier:
        raise AssertionError("model resolution must not run before the test lock")

    monkeypatch.setattr(evaluate_module, "_load_classifier", fail)

    result = CliRunner().invoke(app, ["evaluate", "--split", "test", "--model", "nonexistent"])

    assert result.exit_code == 6
    assert "test split is locked; pass --final to record an audited evaluation" in ANSI_ESCAPE.sub(
        "", result.stderr + result.stdout
    )


def test_missing_model_with_final_exits_four() -> None:
    result = CliRunner().invoke(
        app, ["evaluate", "--split", "test", "--final", "--model", "definitely-missing"]
    )

    output = ANSI_ESCAPE.sub("", result.stderr + result.stdout)
    assert result.exit_code == 4
    assert "model not found" in output
    assert "stc train baseline" in output


def test_dev_evaluation_prints_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(evaluate_module, "_load_examples", lambda *_: make_examples())
    monkeypatch.setattr(evaluate_module, "_load_classifier", lambda *_: FakeClassifier())

    result = CliRunner().invoke(app, ["evaluate", "--split", "dev", "--model", "fake", "--json"])

    assert result.exit_code == 0, result.stderr
    payload = json.loads(ANSI_ESCAPE.sub("", result.stdout))
    assert payload["split"] == "dev"
    assert payload["macro_f1"] == pytest.approx(1.0 / 3.0, abs=1e-3)
    assert payload["accuracy"] == pytest.approx(0.5)
    assert len(payload["errors"]) == 1


def test_dev_evaluation_writes_a_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(evaluate_module, "_load_examples", lambda *_: make_examples())
    monkeypatch.setattr(evaluate_module, "_load_classifier", lambda *_: FakeClassifier())
    out = tmp_path / "reports"

    result = CliRunner().invoke(
        app, ["evaluate", "--split", "dev", "--model", "fake", "--out", str(out)]
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["n_examples"] == 2


def test_final_evaluation_writes_exactly_one_audit_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(evaluate_module, "_load_examples", lambda *_: make_examples())
    monkeypatch.setattr(evaluate_module, "_load_classifier", lambda *_: FakeClassifier())
    log_path = tmp_path / "test_access_log.jsonl"
    monkeypatch.setattr(evaluate_module, "_access_log_path", lambda: log_path)

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--split",
            "test",
            "--final",
            "--model",
            "fake",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stderr
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["n_examples"] == 2
    assert "stc evaluate" in entry["command"]


def test_invalid_split_is_a_usage_error() -> None:
    result = CliRunner().invoke(app, ["evaluate", "--split", "bogus", "--model", "x"])

    assert result.exit_code == 2


def test_train_split_is_not_evaluable() -> None:
    result = CliRunner().invoke(app, ["evaluate", "--split", "train", "--model", "x"])

    assert result.exit_code == 2


def test_dev_evaluation_with_robustness_adds_the_perturbation_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(evaluate_module, "_load_examples", lambda *_: make_examples())
    monkeypatch.setattr(evaluate_module, "_load_classifier", lambda *_: FakeClassifier())
    out = tmp_path / "reports"

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--split",
            "dev",
            "--model",
            "fake",
            "--robustness",
            "--out",
            str(out),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(ANSI_ESCAPE.sub("", result.stdout))
    perturbations = payload["robustness"]["perturbations"]
    assert [row["name"] for row in perturbations] == [
        "no_accents",
        "uppercase",
        "typos",
        "chat_abbrev",
        "regionalize:mx",
        "regionalize:ec",
        "regionalize:ar",
    ]
    assert all("n_affected" in row for row in perturbations)
    written = json.loads((out / "robustness.json").read_text(encoding="utf-8"))
    assert written["split"] == "dev"
