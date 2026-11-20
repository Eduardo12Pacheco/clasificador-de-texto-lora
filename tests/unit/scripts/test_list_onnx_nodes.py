"""The ONNX node lister helps validate the ``mlp_only`` exclusion patterns."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "list_onnx_nodes.py"
TINY_MODEL = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle" / "model.onnx"


def test_lists_nodes_of_the_tiny_model() -> None:
    pytest.importorskip("onnx")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(TINY_MODEL)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Gather" in result.stdout
    assert "MatMul" in result.stdout


def test_missing_model_fails_cleanly() -> None:
    pytest.importorskip("onnx")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "/nonexistent/model.onnx"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
