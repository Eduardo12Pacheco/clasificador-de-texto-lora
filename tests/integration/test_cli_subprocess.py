"""The CLI works as ``python -m stc`` in a real subprocess (S13)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"


def _run(
    args: list[str], *, env: dict[str, str] | None = None, **kwargs: object
) -> subprocess.CompletedProcess[str]:
    # Overlay on the inherited environment: on Windows a child without SystemRoot
    # cannot initialise Winsock (WinError 10106) when it imports asyncio.
    return subprocess.run(
        [sys.executable, "-m", "stc", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **(env or {})},
        **kwargs,
    )


def test_module_entrypoint_prints_the_version() -> None:
    result = _run(["--version"])

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stc 0.1.0"


def test_module_entrypoint_predicts_json_from_stdin(tmp_path: Path) -> None:
    home = tmp_path / "home"
    packed = _run(
        [
            "bundle",
            "pack",
            "--bundle",
            str(TINY_BUNDLE),
            "--out",
            str(tmp_path / "tiny.zip"),
        ]
    )
    assert packed.returncode == 0, packed.stderr
    pulled = _run(
        ["model", "pull", "--from-file", str(tmp_path / "tiny.zip")],
        env={"STC_HOME": str(home)},
    )
    assert pulled.returncode == 0, pulled.stderr

    result = _run(
        ["predict", "-", "--json"],
        input="apaga la luz\n",
        env={"STC_HOME": str(home), "STC_MODEL": "stc-tiny-test"},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["model"]["id"] == "stc-tiny-test"
    assert len(payload["predictions"]) == 1
