"""Composition root: the only place that wires ports to concrete adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stc import __version__, config
from stc.adapters.data.fixture_source import FixtureRepository
from stc.adapters.data.jsonl_repository import JsonlDatasetRepository
from stc.adapters.data.massive_archive import PINNED_SHA256, PINNED_URL, MassiveArchiveSource
from stc.adapters.data.massive_parser import MassiveParser
from stc.application.ports import DatasetRepository, RawDatasetSource
from stc.application.use_cases.prepare_data import PrepareDataPipeline
from stc.config import Settings


@dataclass(frozen=True, slots=True)
class DataServices:
    """Everything the ``stc data`` commands need, wired once."""

    settings: Settings
    raw_dir: Path
    processed_dir: Path
    source: RawDatasetSource
    repository: DatasetRepository
    pipeline: PrepareDataPipeline
    fixtures: FixtureRepository


def build_settings() -> Settings:
    """Load settings from the ``STC_*`` environment."""
    return config.Settings()


def build_data_services(
    settings: Settings | None = None,
    *,
    url: str | None = None,
    sha256: str | None = None,
) -> DataServices:
    """Wire the data adapters for one CLI invocation."""
    resolved = settings if settings is not None else build_settings()
    data_dir = resolved.resolve_data_dir()
    raw_dir = data_dir / "raw" / "massive-1.0"
    processed_dir = data_dir / "processed" / "massive-1.0"
    source = MassiveArchiveSource(
        url=url if url is not None else PINNED_URL,
        sha256=sha256 if sha256 is not None else PINNED_SHA256,
        offline=resolved.offline,
    )
    repository = JsonlDatasetRepository(processed_dir)
    pipeline = PrepareDataPipeline(
        source=source,
        parser=MassiveParser(),
        repository=repository,
        raw_dir=raw_dir,
        stc_version=__version__,
    )
    return DataServices(
        settings=resolved,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        source=source,
        repository=repository,
        pipeline=pipeline,
        fixtures=FixtureRepository(),
    )
