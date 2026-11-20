"""CLI tests for ``stc predict`` with the committed tiny bundle."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stc.adapters.cli.app import app

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
runner = CliRunner()


def _plain(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def _installed_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    packed = runner.invoke(
        app,
        [
            "bundle",
            "pack",
            "--bundle",
            str(TINY_BUNDLE),
            "--out",
            str(tmp_path / "tiny.zip"),
        ],
    )
    assert packed.exit_code == 0, packed.output
    pulled = runner.invoke(
        app,
        ["model", "pull", "--from-file", str(tmp_path / "tiny.zip")],
        env={"STC_HOME": str(home)},
    )
    assert pulled.exit_code == 0, pulled.output
    return home


def _env(home: Path) -> dict[str, str]:
    return {"STC_HOME": str(home), "STC_MODEL": "stc-tiny-test"}


def test_predict_from_an_argument_emits_the_documented_json(tmp_path: Path) -> None:
    home = _installed_home(tmp_path)

    result = runner.invoke(
        app,
        ["predict", "enciende la luz", "--json"],
        env=_env(home),
    )

    assert result.exit_code == 0, _plain(result.stderr)
    payload = json.loads(_plain(result.stdout))
    assert payload["schema_version"] == 1
    assert payload["model"]["id"] == "stc-tiny-test"
    assert payload["model"]["temperature"] == 1.5
    prediction = payload["predictions"][0]
    assert {"intent", "scenario", "confidence", "needs_review", "top_k"} <= set(prediction)
    assert len(prediction["top_k"]) == 3


def test_predict_reads_one_text_per_non_blank_line_from_a_file(tmp_path: Path) -> None:
    home = _installed_home(tmp_path)
    source = tmp_path / "texts.txt"
    source.write_text("a\n\nb\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["predict", "--file", str(source), "--json"],
        env=_env(home),
    )

    assert result.exit_code == 0, _plain(result.stderr)
    payload = json.loads(_plain(result.stdout))
    assert len(payload["predictions"]) == 2
    assert [item["index"] for item in payload["predictions"]] == [0, 1]


def test_predict_reads_stdin_when_the_text_is_a_dash(tmp_path: Path) -> None:
    home = _installed_home(tmp_path)

    result = runner.invoke(
        app,
        ["predict", "-", "--json"],
        input="a\nb\n",
        env=_env(home),
    )

    assert result.exit_code == 0, _plain(result.stderr)
    payload = json.loads(_plain(result.stdout))
    assert len(payload["predictions"]) == 2


def test_predict_without_input_is_a_usage_error(tmp_path: Path) -> None:
    home = _installed_home(tmp_path)

    result = runner.invoke(app, ["predict", "--json"], input="", env=_env(home))

    assert result.exit_code == 2
    assert "no input" in _plain(result.stderr + result.stdout).lower()


def test_predict_with_a_missing_model_exits_four_with_the_pull_hint(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["predict", "hola", "--model", "definitely-missing"],
        env={"STC_HOME": str(tmp_path / "empty-home")},
    )

    output = _plain(result.stderr + result.stdout)
    assert result.exit_code == 4
    assert "stc model pull" in output


def test_predict_renders_a_table_by_default(tmp_path: Path) -> None:
    home = _installed_home(tmp_path)

    result = runner.invoke(app, ["predict", "apaga la luz"], env=_env(home))

    assert result.exit_code == 0, _plain(result.stderr)
    output = _plain(result.stdout)
    assert "intent" in output
    assert "scenario" in output
    assert "confidence" in output
