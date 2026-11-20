"""``stc report final`` writes an auditable, validated final report."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stc.adapters.cli.app import app
from stc.adapters.cli.commands import report as report_module

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
runner = CliRunner()

REQUIRED_FILES = (
    "metrics.json",
    "comparison.json",
    "per_class.csv",
    "confusion_matrix.csv",
    "calibration_test.json",
    "robustness.json",
    "slices.json",
    "errors.json",
)


def _plain(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def test_report_final_writes_every_validated_file_and_one_audit_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    trained = runner.invoke(
        app,
        ["train", "baseline", "--dataset", "fixture", "--quick", "--out", str(baseline)],
    )
    assert trained.exit_code == 0, _plain(trained.stderr)

    bundle = tmp_path / "bundle"
    shutil.copytree(TINY_BUNDLE, bundle)
    calibrated = runner.invoke(
        app,
        [
            "calibrate",
            "--bundle",
            str(bundle),
            "--baseline",
            str(baseline),
            "--dataset",
            "fixture",
            "--reports",
            str(tmp_path / "calibration"),
        ],
    )
    assert calibrated.exit_code == 0, _plain(calibrated.stderr)

    access_log = tmp_path / "test_access_log.jsonl"
    monkeypatch.setattr(report_module, "_access_log_path", lambda: access_log)
    out = tmp_path / "reports" / "final"
    figures = tmp_path / "figures"

    result = runner.invoke(
        app,
        [
            "report",
            "final",
            "--bundle",
            str(bundle),
            "--baseline",
            str(baseline),
            "--out",
            str(out),
            "--figures",
            str(figures),
            "--dataset",
            "fixture",
        ],
        env={"STC_HOME": str(tmp_path / "home")},
    )

    assert result.exit_code == 0, _plain(result.stderr)
    for name in REQUIRED_FILES:
        assert (out / name).is_file(), name
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["verdict"] in {"transformer_better", "baseline_better", "tie"}
    assert comparison["split"] == "test"
    lines = access_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["n_examples"] > 0
    assert entry["model_sha256"] != "unknown"
    assert "stc report final" in entry["command"]

    validated = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_reports.py"),
            "final",
            "--root",
            str(tmp_path / "reports"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validated.returncode == 0, validated.stderr


def test_evaluate_test_without_final_is_locked(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["evaluate", "--split", "test", "--model", "anything"],
        env={"STC_HOME": str(tmp_path)},
    )

    assert result.exit_code == 6
