"""The experiments validator rejects malformed screening and search reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_collect_experiment_reports import VALIDATE, _run

from conftest import write_experiment_reports


def test_valid_experiment_reports_exit_zero(tmp_path: Path) -> None:
    write_experiment_reports(tmp_path)

    result = _run(VALIDATE, "experiments", "--root", str(tmp_path))

    assert result.returncode == 0, result.stderr + result.stdout


def test_unknown_selection_branch_is_rejected(tmp_path: Path) -> None:
    write_experiment_reports(tmp_path)
    path = tmp_path / "selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rule_branch"] = "qwen_wins"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(VALIDATE, "experiments", "--root", str(tmp_path))

    assert result.returncode == 1
    assert "unknown branch" in result.stderr


def test_missing_search_trials_is_named(tmp_path: Path) -> None:
    write_experiment_reports(tmp_path)
    (tmp_path / "search_trials.json").unlink()

    result = _run(VALIDATE, "experiments", "--root", str(tmp_path))

    assert result.returncode == 1
    assert "missing report" in result.stderr


def test_experiments_default_root_is_reports_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_experiment_reports(tmp_path / "reports" / "experiments")
    monkeypatch.chdir(tmp_path)

    result = _run(VALIDATE, "experiments")

    assert result.returncode == 0, result.stderr + result.stdout
