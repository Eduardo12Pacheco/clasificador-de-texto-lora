"""The CLI README image is generated without third-party resources."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "capture_cli_svg.py"
TINY_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"


def test_svg_is_self_contained(tmp_path: Path) -> None:
    out = tmp_path / "cli-predict.svg"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--bundle", str(TINY_BUNDLE), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rendered = out.read_text(encoding="utf-8")
    assert rendered.startswith("<svg")
    assert "Predictions" in rendered
    assert "cdnjs" not in rendered
    assert "Fira Code" not in rendered
