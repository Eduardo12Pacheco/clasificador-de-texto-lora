"""Tests for the ``stc`` command-line entrypoint."""

import pytest
from typer.testing import CliRunner

from stc import __version__
from stc.adapters.cli.app import app

pytestmark = pytest.mark.unit


def test_version_option_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"stc {__version__}"
    assert result.stdout.strip() == "stc 0.1.0"


def test_help_lists_the_application_without_error() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Spanish virtual-assistant intent classifier" in result.stdout
    assert "--version" in result.stdout
