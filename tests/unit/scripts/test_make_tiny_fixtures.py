"""The tiny fixture generator is deterministic and produces a loadable tokenizer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "make_tiny_fixtures.py"


def test_generation_is_byte_identical_on_a_second_run(tmp_path: Path) -> None:
    first = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path), "--tokenizer-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    before = (tmp_path / "tokenizer.json").read_bytes()

    second = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path), "--tokenizer-only"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert second.returncode == 0, second.stderr
    assert (tmp_path / "tokenizer.json").read_bytes() == before


def test_committed_tokenizer_matches_the_generator(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path), "--tokenizer-only"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    committed = REPO_ROOT / "tests" / "fixtures" / "tiny" / "tokenizer.json"
    assert (tmp_path / "tokenizer.json").read_bytes() == committed.read_bytes()


def test_committed_bundle_matches_the_generator(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out",
            str(tmp_path / "tiny"),
            "--bundle-out",
            str(tmp_path / "tiny_bundle"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    committed = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"
    generated = tmp_path / "tiny_bundle"
    for path in sorted(committed.iterdir()):
        assert path.name in {item.name for item in generated.iterdir()}
        assert (generated / path.name).read_bytes() == path.read_bytes(), path.name
    assert (tmp_path / "tiny" / "tokenizer.json").is_file()
