#!/usr/bin/env python3
"""Fail when tracked files are large or look like model weights or bundles."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_TRACKED_BYTES = 1_000_000
WEIGHT_SUFFIXES = frozenset(
    {
        ".onnx",
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
        ".skops",
        ".npy",
        ".npz",
        ".joblib",
        ".zip",
    }
)
ALLOWED_WEIGHT_PREFIXES = ("tests/fixtures/", "src/stc/_fixture/")


def tracked_files(root: Path) -> list[str]:
    """Return every path tracked by git, relative to ``root``."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        text=False,
    )
    return [path for path in result.stdout.decode("utf-8").split("\0") if path]


def offenders(root: Path, paths: list[str]) -> list[str]:
    """Return human-readable violations for the given tracked paths."""
    violations: list[str] = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            continue
        size = path.stat().st_size
        allowed_weight = relative.startswith(ALLOWED_WEIGHT_PREFIXES)
        if size > MAX_TRACKED_BYTES:
            violations.append(f"{relative}: {size} bytes exceeds {MAX_TRACKED_BYTES}")
        if path.suffix.lower() in WEIGHT_SUFFIXES and not allowed_weight:
            violations.append(f"{relative}: weight or archive file outside the fixture allow-list")
    return violations


def main() -> int:
    """Run the check and return a process exit code."""
    root = Path(__file__).resolve().parents[1]
    violations = offenders(root, tracked_files(root))
    if violations:
        print("Tracked files violate the repository artifact policy:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1
    print("OK: no tracked weights or files above 1 MB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
