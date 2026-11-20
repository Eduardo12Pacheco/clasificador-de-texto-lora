#!/usr/bin/env python3
"""Validate GitHub Actions workflows against the repository policy."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

WORKFLOW_DIR = Path(".github/workflows")
FORBIDDEN_REFS = frozenset({"main", "master"})


def _workflow_files(root: Path) -> list[Path]:
    directory = root / WORKFLOW_DIR
    return sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")])


def _check_uses(value: str, *, where: str) -> list[str]:
    if value.startswith("./"):
        return []
    if "@" not in value:
        return [f"{where}: action reference {value!r} is not pinned to a ref"]
    ref = value.rsplit("@", maxsplit=1)[1].strip()
    if not ref:
        return [f"{where}: action reference {value!r} has an empty ref"]
    if ref in FORBIDDEN_REFS:
        return [f"{where}: action reference {value!r} must not track {ref}"]
    return []


def validate_workflow(path: Path) -> list[str]:
    """Return policy violations found in one workflow file."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"{path}: invalid YAML: {exc}"]

    if not isinstance(document, dict):
        return [f"{path}: workflow must be a mapping"]

    errors: list[str] = []
    if "permissions" not in document:
        errors.append(f"{path}: missing workflow-level permissions")

    jobs = document.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        errors.append(f"{path}: workflow defines no jobs")
        return errors

    for job_name, job in jobs.items():
        where = f"{path}:{job_name}"
        if not isinstance(job, dict):
            errors.append(f"{where}: job must be a mapping")
            continue
        if "runs-on" not in job:
            errors.append(f"{where}: job is missing runs-on")
        steps = job.get("steps", [])
        if not isinstance(steps, list):
            errors.append(f"{where}: steps must be a list")
            continue
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"{where}: step {index} must be a mapping")
                continue
            uses = step.get("uses")
            if isinstance(uses, str):
                errors.extend(_check_uses(uses, where=f"{where}: step {index}"))
            elif uses is not None:
                errors.append(f"{where}: step {index} uses must be a string")
    return errors


def main() -> int:
    """Validate every workflow and return a process exit code."""
    root = Path(__file__).resolve().parents[1]
    files = _workflow_files(root)
    if not files:
        print("No workflow files found.", file=sys.stderr)
        return 1

    errors: list[str] = []
    for path in files:
        errors.extend(validate_workflow(path.relative_to(root)))

    if errors:
        print("Workflow validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"OK: {len(files)} workflow file(s) validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
