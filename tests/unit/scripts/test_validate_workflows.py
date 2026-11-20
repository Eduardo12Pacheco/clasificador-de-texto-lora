"""Workflow validation enforces the T13 CI/CD shape."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "validate_workflows.py"

CI = """
name: CI
on: {push: {branches: [main]}}
permissions: {contents: read}
jobs:
  quality: {runs-on: ubuntu-latest, steps: [{uses: actions/checkout@v4}]}
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        include: [{os: ubuntu-latest, python: "3.12"}, {os: ubuntu-latest, python: "3.13"}]
    steps: [{run: "pytest"}]
  test-torch: {runs-on: ubuntu-latest, steps: [{run: "pytest --cov-fail-under=85"}]}
  e2e-ui: {runs-on: ubuntu-latest, steps: [{run: "pytest -m e2e"}]}
  smoke-install:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    steps: [{run: "bash scripts/smoke_install.sh"}]
  security: {runs-on: ubuntu-latest, steps: [{run: "gitleaks git; uvx pip-audit"}]}
  docs: {runs-on: ubuntu-latest, steps: [{run: "pytest tests/docs"}]}
"""

RELEASE = """
name: Release
on: {push: {tags: ["v*"]}}
permissions: {contents: write}
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - {uses: actions/checkout@v4}
      - {run: "uvx git-cliff --latest -o RELEASE_NOTES.md"}
      - {run: "sha256sum dist/* > SHA256SUMS"}
      - {run: "gh release create \\"$TAG\\" dist/* SHA256SUMS"}
"""

VERIFY = """
name: Verify release
on:
  release: {types: [published]}
  workflow_dispatch:
    inputs:
      tag: {description: Tag, required: true}
permissions: {contents: read}
jobs:
  verify:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    steps:
      - {run: "stc model pull && stc model verify && stc doctor"}
      - {run: "python scripts/check_smoke_phrases.py tests/fixtures/smoke_phrases.json"}
"""


def _module():
    spec = importlib.util.spec_from_file_location("validate_workflows", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo(tmp_path: Path, **overrides: str) -> Path:
    directory = tmp_path / ".github" / "workflows"
    directory.mkdir(parents=True)
    documents = {"ci.yml": CI, "release.yml": RELEASE, "verify-release.yml": VERIFY}
    documents.update(overrides)
    for name, text in documents.items():
        (directory / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_complete_workflow_set_passes(tmp_path: Path) -> None:
    assert _module().validate_repository(_repo(tmp_path)) == []


def test_missing_workflow_is_reported(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / ".github" / "workflows" / "release.yml").unlink()

    errors = _module().validate_repository(root)

    assert any("release.yml: required workflow is missing" in error for error in errors)


def test_ci_missing_job_is_reported(tmp_path: Path) -> None:
    docs_job = '  docs: {runs-on: ubuntu-latest, steps: [{run: "pytest tests/docs"}]}\n'
    broken = CI.replace(docs_job, "")

    errors = _module().validate_repository(_repo(tmp_path, **{"ci.yml": broken}))

    assert any("missing required jobs: docs" in error for error in errors)


def test_ci_requires_coverage_gate_and_three_os(tmp_path: Path) -> None:
    broken = CI.replace("--cov-fail-under=85", "").replace("macos-latest", "linux-arm")

    errors = _module().validate_repository(_repo(tmp_path, **{"ci.yml": broken}))

    assert any("coverage gate" in error for error in errors)
    assert any("test matrix is missing macos-latest" in error for error in errors)


def test_release_requires_tag_trigger_and_write_permission(tmp_path: Path) -> None:
    broken = RELEASE.replace('tags: ["v*"]', "branches: [main]").replace(
        "contents: write", "contents: read"
    )

    errors = _module().validate_repository(_repo(tmp_path, **{"release.yml": broken}))

    assert any("v* tags" in error for error in errors)
    assert any("contents: write" in error for error in errors)


def test_unpinned_and_floating_actions_are_rejected(tmp_path: Path) -> None:
    broken = CI.replace("actions/checkout@v4", "actions/checkout@main")

    errors = _module().validate_repository(_repo(tmp_path, **{"ci.yml": broken}))

    assert any("must not track main" in error for error in errors)


def test_pull_request_target_is_rejected(tmp_path: Path) -> None:
    broken = CI.replace("on: {push: {branches: [main]}}", "on: {pull_request_target: {}}")

    errors = _module().validate_repository(_repo(tmp_path, **{"ci.yml": broken}))

    assert any("pull_request_target" in error for error in errors)
