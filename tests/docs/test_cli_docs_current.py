"""The committed CLI reference is generated from the live Typer app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "gen_cli_docs.py"
DOC = REPO_ROOT / "docs" / "CLI.md"


def test_generated_cli_reference_matches_the_committed_file(tmp_path: Path) -> None:
    generated = tmp_path / "CLI.md"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(generated)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert generated.read_bytes() == DOC.read_bytes()


def test_cli_reference_covers_every_command_group() -> None:
    text = DOC.read_text(encoding="utf-8")

    for command in (
        "stc data",
        "stc train",
        "stc export",
        "stc calibrate",
        "stc bundle",
        "stc model",
        "stc evaluate",
        "stc report",
        "stc predict",
        "stc benchmark",
        "stc serve",
        "stc doctor",
    ):
        assert f"## `{command}`" in text
    for code in range(7):
        assert f"| {code} |" in text
    predict_block = text.split("## `stc predict`", maxsplit=1)[1].split("## `", maxsplit=1)[0]
    assert "--top-k" in predict_block
    assert "Classify Spanish virtual-assistant requests" in predict_block
