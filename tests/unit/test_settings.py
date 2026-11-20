"""Tests for typed settings and structured-logging setup."""

from __future__ import annotations

import structlog

from stc.config import Settings
from stc.logging import configure_logging


def test_defaults_resolve_under_the_platform_data_dir(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("STC_HOME", str(tmp_path))
    for name in ("STC_DATA_DIR", "STC_API_PORT", "STC_OFFLINE", "STC_THREADS"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.resolve_home() == tmp_path
    assert settings.resolve_data_dir() == tmp_path / "data"
    assert 19000 <= settings.api_port <= 19999
    assert 1 <= settings.threads <= 4


def test_data_dir_can_be_overridden(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("STC_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STC_DATA_DIR", str(tmp_path / "elsewhere"))

    settings = Settings()

    assert settings.resolve_data_dir() == tmp_path / "elsewhere"


def test_environment_booleans_and_integers_are_parsed(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("STC_OFFLINE", "1")
    monkeypatch.setenv("STC_SEED", "42")
    monkeypatch.setenv("STC_THREADS", "2")

    settings = Settings()

    assert settings.offline is True
    assert settings.seed == 42
    assert settings.threads == 2


def test_log_format_resolution(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("STC_LOG_FORMAT", raising=False)
    settings = Settings()
    assert settings.resolve_log_format() in {"console", "json"}

    monkeypatch.setenv("STC_LOG_FORMAT", "json")
    assert Settings().resolve_log_format() == "json"


def test_json_logging_emits_one_object(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging("INFO", "json")

    structlog.get_logger("test").info("classified", n_texts=2)

    captured = capsys.readouterr()
    assert '"event": "classified"' in captured.err
    assert '"n_texts": 2' in captured.err


def test_console_logging_is_human_readable(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging("INFO", "console")

    structlog.get_logger("test").info("prepared")

    captured = capsys.readouterr()
    assert "prepared" in captured.err
