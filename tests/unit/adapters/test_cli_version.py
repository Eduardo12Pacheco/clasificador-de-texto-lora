"""Tests for the ``stc`` command-line entrypoint."""

import re

import pytest
from typer.testing import CliRunner

from stc import __version__
from stc.adapters.cli.app import app

pytestmark = pytest.mark.unit

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    return ANSI_ESCAPE.sub("", output)


def test_version_option_prints_package_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"stc {__version__}"
    assert result.stdout.strip() == "stc 0.1.0"


def test_help_lists_the_application_without_error() -> None:
    result = CliRunner().invoke(app, ["--help"])
    output = _plain(result.stdout)

    assert result.exit_code == 0
    assert "Spanish virtual-assistant intent classifier" in output
    assert "--version" in output
