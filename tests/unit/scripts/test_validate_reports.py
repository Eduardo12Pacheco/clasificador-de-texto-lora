"""``validate_reports.py`` accepts complete reports and names real defects."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import write_all_reports, write_final_reports

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "validate_reports.py"


def _run(kind: str, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), kind, "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_valid_final_reports_exit_zero(tmp_path: Path) -> None:
    write_final_reports(tmp_path)

    result = _run("final", tmp_path)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "OK final" in result.stdout


def test_unknown_verdict_is_rejected(tmp_path: Path) -> None:
    write_final_reports(tmp_path)
    path = tmp_path / "final" / "comparison.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["verdict"] = "maybe"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run("final", tmp_path)

    assert result.returncode == 1
    assert "unknown verdict" in result.stderr


def test_inverted_interval_is_rejected(tmp_path: Path) -> None:
    write_final_reports(tmp_path)
    path = tmp_path / "final" / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["models"]["lora"]["ci"]["macro_f1"]["lower"] = 0.99
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run("final", tmp_path)

    assert result.returncode == 1
    assert "estimate must lie" in result.stderr


def test_missing_report_is_named(tmp_path: Path) -> None:
    write_final_reports(tmp_path)
    (tmp_path / "final" / "errors.json").unlink()

    result = _run("final", tmp_path)

    assert result.returncode == 1
    assert "missing report" in result.stderr


def test_valid_all_reports_exit_zero(tmp_path: Path) -> None:
    write_all_reports(tmp_path)

    result = _run("all", tmp_path)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "OK all" in result.stdout
