#!/usr/bin/env python3
"""Validate GitHub Actions workflows against the repository release policy (S19)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

WORKFLOW_DIR = Path(".github/workflows")
FORBIDDEN_REFS = frozenset({"main", "master"})
EXPECTED_WORKFLOWS = ("ci.yml", "release.yml", "verify-release.yml")
CI_REQUIRED_JOBS = frozenset(
    {"quality", "test", "test-torch", "e2e-ui", "smoke-install", "security", "docs"}
)


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


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _check_generic(document: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    if "permissions" not in document:
        errors.append(f"{path}: missing workflow-level permissions")
    triggers = document.get("on", document.get(True))
    if "pull_request_target" in _as_mapping(triggers) or triggers == "pull_request_target":
        errors.append(f"{path}: pull_request_target is not allowed")

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
        if not isinstance(steps, list) or not steps:
            errors.append(f"{where}: job defines no steps")
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
            if "run" in step and not isinstance(step["run"], str):
                errors.append(f"{where}: step {index} run must be a string")
    return errors


def _check_ci(document: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    jobs = _as_mapping(document.get("jobs"))
    missing = CI_REQUIRED_JOBS - set(jobs)
    if missing:
        errors.append(f"{path}: missing required jobs: {', '.join(sorted(missing))}")
    permissions = _as_mapping(document.get("permissions"))
    if permissions.get("contents") != "read":
        errors.append(f"{path}: CI must request contents: read")

    test_job = _as_mapping(jobs.get("test"))
    matrix = _as_mapping(_as_mapping(test_job.get("strategy")).get("matrix"))
    include = matrix.get("include")
    entries = (
        [entry for entry in include if isinstance(entry, dict)] if isinstance(include, list) else []
    )
    os_matrix = matrix.get("os")
    if isinstance(os_matrix, list):
        for runner in ("ubuntu-latest", "windows-latest", "macos-latest"):
            if runner not in os_matrix:
                errors.append(f"{path}: test matrix is missing {runner}")
        if not any(entry.get("python") == "3.13" for entry in entries):
            errors.append(f"{path}: test matrix is missing the Python 3.13 ubuntu entry")

    torch = _as_mapping(jobs.get("test-torch"))
    torch_text = str(torch)
    if "--cov-fail-under=85" not in torch_text:
        errors.append(f"{path}: test-torch must enforce the coverage gate")

    smoke_job = _as_mapping(jobs.get("smoke-install"))
    smoke_matrix = _as_mapping(_as_mapping(smoke_job.get("strategy")).get("matrix"))
    smoke_os = smoke_matrix.get("os")
    if isinstance(smoke_os, list):
        for runner in ("ubuntu-latest", "windows-latest", "macos-latest"):
            if runner not in smoke_os:
                errors.append(f"{path}: smoke-install matrix is missing {runner}")

    security_text = str(_as_mapping(jobs.get("security")))
    for needle in ("gitleaks", "pip-audit"):
        if needle not in security_text:
            errors.append(f"{path}: security job must run {needle}")
    return errors


def _check_release(document: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    triggers = _as_mapping(document.get("on", document.get(True)))
    push = _as_mapping(triggers.get("push"))
    tags = push.get("tags")
    if not (isinstance(tags, list) and any(str(tag).startswith("v") for tag in tags)):
        errors.append(f"{path}: release must trigger on v* tags")
    permissions = _as_mapping(document.get("permissions"))
    if permissions.get("contents") != "write":
        errors.append(f"{path}: release must request contents: write")
    text = str(document)
    for needle in ("git-cliff", "gh release create", "SHA256SUMS"):
        if needle not in text:
            errors.append(f"{path}: release pipeline is missing {needle}")
    return errors


def _check_verify_release(document: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    triggers = _as_mapping(document.get("on", document.get(True)))
    release = _as_mapping(triggers.get("release"))
    if "published" not in str(release):
        errors.append(f"{path}: verify-release must run when a release is published")
    dispatch = _as_mapping(triggers.get("workflow_dispatch"))
    inputs = _as_mapping(dispatch.get("inputs"))
    if "tag" not in inputs:
        errors.append(f"{path}: verify-release workflow_dispatch needs a tag input")
    text = str(document)
    for needle in ("stc model pull", "stc model verify", "smoke_phrases.json", "stc doctor"):
        if needle not in text:
            errors.append(f"{path}: verify-release is missing {needle}")
    return errors


def validate_workflow(path: Path) -> list[str]:
    """Return policy violations found in one workflow file."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"{path}: invalid YAML: {exc}"]

    if not isinstance(document, dict):
        return [f"{path}: workflow must be a mapping"]

    errors = _check_generic(document, path)
    if path.name == "ci.yml":
        errors.extend(_check_ci(document, path))
    elif path.name == "release.yml":
        errors.extend(_check_release(document, path))
    elif path.name == "verify-release.yml":
        errors.extend(_check_verify_release(document, path))
    return errors


def validate_repository(root: Path) -> list[str]:
    """Validate the expected workflow set and every workflow's policy."""
    errors: list[str] = []
    directory = root / WORKFLOW_DIR
    for name in EXPECTED_WORKFLOWS:
        if not (directory / name).is_file():
            errors.append(f"{WORKFLOW_DIR / name}: required workflow is missing")
    files = _workflow_files(root)
    if not files:
        errors.append("No workflow files found.")
    for path in files:
        errors.extend(validate_workflow(path))
    return errors


def main() -> int:
    """Validate every workflow and return a process exit code."""
    root = Path(__file__).resolve().parents[1]
    errors = validate_repository(root)
    if errors:
        print("Workflow validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"OK: {len(_workflow_files(root))} workflow file(s) validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
