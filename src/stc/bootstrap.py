"""Composition root: the only place that wires ports to concrete adapters."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from stc import __version__, config
from stc.adapters.data.fixture_source import FixtureRepository
from stc.adapters.data.jsonl_repository import JsonlDatasetRepository
from stc.adapters.data.massive_archive import PINNED_SHA256, PINNED_URL, MassiveArchiveSource
from stc.adapters.data.massive_parser import MassiveParser
from stc.adapters.models.skops_store import SkopsBaselineStore
from stc.adapters.models.tfidf_linear import TfidfLinearTrainer
from stc.adapters.reporting.json_writer import JsonReportWriter
from stc.adapters.storage.access_log import JsonlAccessLog
from stc.application.ports import (
    Classifier,
    ClassifierTrainer,
    DatasetRepository,
    ExperimentTracker,
    HyperparameterSearcher,
    RawDatasetSource,
    ScreeningRunner,
)
from stc.application.services.tracker_factory import TrackerFactory
from stc.application.use_cases.prepare_data import PrepareDataPipeline
from stc.application.use_cases.train_baseline import TrainBaseline
from stc.config import Settings
from stc.domain.errors import ArtifactNotFoundError, InvalidInputError, MissingExtraError
from stc.domain.examples import Example


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


@dataclass(frozen=True, slots=True)
class BaselineServices:
    """Everything ``stc train baseline`` needs, wired once."""

    repository: DatasetRepository
    use_case: TrainBaseline


def build_baseline_services(
    settings: Settings | None = None,
    *,
    dataset: str = "massive",
) -> BaselineServices:
    """Wire the baseline trainer, store and report writer for one CLI invocation."""
    data = build_data_services(settings)
    if dataset == "fixture":
        repository: DatasetRepository = data.fixtures
    elif dataset == "massive":
        repository = data.repository
    else:
        raise InvalidInputError(f"unknown dataset: {dataset!r}; expected massive or fixture")
    return BaselineServices(
        repository=repository,
        use_case=TrainBaseline(
            TfidfLinearTrainer(),
            SkopsBaselineStore(),
            build_report_writer(),
        ),
    )


def build_report_writer() -> JsonReportWriter:
    """Wire the deterministic JSON report writer for a CLI invocation."""
    return JsonReportWriter()


def build_access_log(path: Path) -> JsonlAccessLog:
    """Wire the append-only test-access audit log for a CLI invocation."""
    return JsonlAccessLog(path)


def load_classifier(model: str) -> Classifier:
    """Resolve a model path or registry name to a classifier (training task adds kinds)."""
    path = Path(model)
    if not path.exists():
        raise ArtifactNotFoundError(
            f"model not found: {model}; run: stc train baseline to create one"
        )
    module = importlib.import_module("stc.adapters.models.tfidf_linear")
    loader = cast(Callable[[Path], Classifier], module.load_baseline)
    return loader(path)


TRAIN_EXTRA_MESSAGE = (
    "training needs the 'train' extra: uv tool install 'spanish-text-classifier-lora[train]'"
)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require_train_extra() -> None:
    """Fail with the documented message when optional training libraries are absent."""
    if not all(_module_available(name) for name in ("torch", "transformers", "peft")):
        raise MissingExtraError(TRAIN_EXTRA_MESSAGE)


def build_training_repository(
    settings: Settings | None = None,
    *,
    dataset: str = "massive",
) -> DatasetRepository:
    """Return the guarded processed or fixture repository."""
    data = build_data_services(settings)
    if dataset == "fixture":
        return data.fixtures
    if dataset == "massive":
        return data.repository
    raise InvalidInputError(f"unknown dataset: {dataset!r}; expected massive or fixture")


def build_tracker_factory(settings: Settings | None = None) -> TrackerFactory:
    """Wire the JSONL/MLflow/Null tracker adapters behind the pure factory."""
    resolved = settings if settings is not None else build_settings()

    def jsonl(run_dir: Path) -> ExperimentTracker:
        module = importlib.import_module("stc.adapters.tracking.jsonl_tracker")
        tracker = cast(Callable[[Path], ExperimentTracker], module.JsonlTracker)
        return tracker(run_dir / "tracker.jsonl")

    def mlflow(run_dir: Path) -> ExperimentTracker:
        del run_dir
        module = importlib.import_module("stc.adapters.tracking.mlflow_tracker")
        tracker = cast(
            Callable[[str], ExperimentTracker],
            module.MlflowTracker,
        )
        return tracker(resolved.mlflow_uri)

    def null(run_dir: Path) -> ExperimentTracker:
        del run_dir
        module = importlib.import_module("stc.adapters.tracking.null_tracker")
        tracker = cast(Callable[[], ExperimentTracker], module.NullTracker)
        return tracker()

    def composite(children: Sequence[ExperimentTracker]) -> ExperimentTracker:
        module = importlib.import_module("stc.adapters.tracking.composite_tracker")
        builder = cast(
            Callable[[Sequence[ExperimentTracker]], ExperimentTracker],
            module.CompositeTracker,
        )
        return builder(children)

    return TrackerFactory(
        {"jsonl": jsonl, "mlflow": mlflow, "null": null},
        composite_builder=composite,
    )


def build_trainer(
    method: str = "lora",
    *,
    device: object | None = None,
    save_artifacts: bool = True,
    on_epoch_end: Callable[[int, float], None] | None = None,
) -> ClassifierTrainer:
    """Wire a LoRA or full fine-tuning trainer (torch is imported lazily)."""
    require_train_extra()
    module = importlib.import_module("stc.adapters.training.trainer")
    name = "LoraTrainer" if method == "lora" else "FullFtTrainer"
    factory = cast(Callable[..., ClassifierTrainer], getattr(module, name))
    return factory(
        save_artifacts=save_artifacts,
        on_epoch_end=on_epoch_end,
        device=device,
    )


def build_searcher(
    train: Sequence[Example],
    dev: Sequence[Example],
    out_dir: Path,
    base_config: Mapping[str, object],
    *,
    tracker_factory: Callable[[Path], ExperimentTracker],
    device: object | None = None,
) -> HyperparameterSearcher:
    """Wire the Optuna searcher lazily so the runtime stays torch-free."""
    require_train_extra()
    module = importlib.import_module("stc.adapters.training.search")
    factory = cast(Callable[..., HyperparameterSearcher], module.OptunaSearcher)
    return factory(
        train,
        dev,
        out_dir,
        base_config,
        tracker_factory=tracker_factory,
        device=device,
    )


def build_screener(
    train: Sequence[Example],
    dev: Sequence[Example],
    out_dir: Path,
    config: Mapping[str, object],
    *,
    tracker_factory: Callable[[Path], ExperimentTracker],
    device: object | None = None,
) -> ScreeningRunner:
    """Wire the backbone screening runner lazily."""
    require_train_extra()
    module = importlib.import_module("stc.adapters.training.screening")
    screening_config = module.ScreeningConfig.from_mapping(config)
    factory = cast(Callable[..., ScreeningRunner], module.ScreeningRunner)
    return factory(
        train,
        dev,
        out_dir,
        screening_config,
        tracker_factory=tracker_factory,
        device=device,
    )


def repository_identity() -> tuple[str, str]:
    """Return ``(git_sha, uv_lock_sha256)`` without failing outside a checkout."""
    git_sha = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        git_sha = result.stdout.strip() or "unknown"
    except OSError:
        pass
    lock = Path("uv.lock")
    try:
        lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
    except OSError:
        lock_sha = "unknown"
    return git_sha, lock_sha
