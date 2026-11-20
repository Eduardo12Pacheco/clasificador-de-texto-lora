"""Package metadata stays consistent between the project and the installed module."""

import importlib.metadata
import tomllib
from pathlib import Path

import pytest

import stc

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_metadata() -> dict[str, object]:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        document = tomllib.load(handle)
    project = document["project"]
    assert isinstance(project, dict)
    return project


def test_version_matches_installed_distribution() -> None:
    project = _project_metadata()

    assert project["name"] == "spanish-text-classifier-lora"
    assert project["version"] == "0.1.0"
    assert stc.__version__ == "0.1.0"
    assert importlib.metadata.version("spanish-text-classifier-lora") == stc.__version__


def test_console_script_is_declared() -> None:
    project = _project_metadata()
    scripts = project["scripts"]

    assert isinstance(scripts, dict)
    assert scripts["stc"] == "stc.adapters.cli.app:main"
