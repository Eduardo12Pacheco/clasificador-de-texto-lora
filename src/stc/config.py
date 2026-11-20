"""Typed application settings loaded from ``STC_*`` environment variables."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import platformdirs
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every runtime knob documented in decision D15, plus the L5 port range."""

    model_config = SettingsConfigDict(
        env_prefix="STC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    home: Path | None = None
    data_dir: Path | None = None
    model: str = "stc-es-intent"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] | None = None
    tracker: str = "jsonl"
    mlflow_uri: str = "sqlite:///artifacts/mlflow.db"
    threads: int = Field(default_factory=lambda: max(1, min(4, os.cpu_count() or 1)))
    offline: bool = False
    seed: int = 13
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=19080, ge=19000, le=19999)
    api_max_batch: int = Field(default=64, ge=1)
    api_max_chars: int = Field(default=512, ge=1)

    def resolve_home(self) -> Path:
        """Return ``STC_HOME`` or the platform data directory."""
        return self.home if self.home is not None else Path(platformdirs.user_data_dir("stc"))

    def resolve_data_dir(self) -> Path:
        """Return ``STC_DATA_DIR`` or ``<home>/data``."""
        return self.data_dir if self.data_dir is not None else self.resolve_home() / "data"

    def resolve_log_format(self) -> Literal["console", "json"]:
        """Return the configured format, defaulting by TTY detection."""
        if self.log_format is not None:
            return self.log_format
        return "console" if sys.stderr.isatty() else "json"
