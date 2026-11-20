"""Every report script is executable through the repository interpreter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = [
    "lexicon_coverage.py",
    "validate_reports.py",
    "render_results.py",
    "render_model_card.py",
    "check_error_analysis.py",
    "collect_experiment_reports.py",
    "make_tiny_fixtures.py",
]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_help_exits_zero(name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / name), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
