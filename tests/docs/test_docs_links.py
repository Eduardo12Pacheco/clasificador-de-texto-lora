"""Every relative Markdown link and heading anchor must resolve."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check_links.py"


def test_every_markdown_link_resolves() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


def test_documentation_suite_is_checked() -> None:
    document = REPO_ROOT / "docs" / "ARCHITECTURE.md"
    assert document.is_file()
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--file", "docs/ARCHITECTURE.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "1 Markdown file(s) checked" in result.stdout
