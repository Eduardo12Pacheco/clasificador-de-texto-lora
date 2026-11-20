"""The T9 CLI surface: global options, report/serve and predict wiring."""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from stc.adapters.cli.app import app
from stc.adapters.cli.commands import serve as serve_module

pytestmark = pytest.mark.unit

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
runner = CliRunner()


def _plain(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def test_help_lists_the_completed_command_surface_and_global_options() -> None:
    result = runner.invoke(app, ["--help"])
    output = _plain(result.stdout)

    assert result.exit_code == 0
    for command in ("predict", "serve", "report"):
        assert command in output
    for option in ("--debug", "--quiet", "--log-format"):
        assert option in output


def test_serve_starts_uvicorn_with_the_requested_host_and_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(application: object, **kwargs: object) -> None:
        calls.append({"application": application, **kwargs})

    monkeypatch.setattr(serve_module.uvicorn, "run", fake_run)

    result = runner.invoke(app, ["serve", "--host", "127.0.0.1", "--port", "8765"])

    assert result.exit_code == 0, _plain(result.stderr)
    assert calls
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 8765


def test_serve_warns_when_the_host_is_not_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve_module.uvicorn, "run", lambda *args, **kwargs: None)

    result = runner.invoke(
        app,
        ["serve", "--host", "0.0.0.0", "--port", "8765"],  # noqa: S104
    )

    assert result.exit_code == 0, _plain(result.stderr)
    assert "loopback" in _plain(result.stdout + result.stderr).lower()


def test_invalid_log_format_is_a_usage_error() -> None:
    result = runner.invoke(app, ["--log-format", "xml", "doctor", "--json"])

    assert result.exit_code == 2
