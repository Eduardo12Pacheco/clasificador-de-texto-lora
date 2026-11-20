"""Composition root: the only place that wires ports to concrete adapters."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from fastapi import FastAPI

from stc import __version__, config
from stc.adapters.data.fixture_source import FixtureRepository
from stc.adapters.data.jsonl_repository import JsonlDatasetRepository
from stc.adapters.data.massive_archive import PINNED_SHA256, PINNED_URL, MassiveArchiveSource
from stc.adapters.data.massive_parser import MassiveParser
from stc.adapters.hub.github_release import GithubReleaseHub
from stc.adapters.hub.registry import ModelRegistry, RegistryEntry
from stc.adapters.inference.bundle_io import BundleArchiveAdapter, read_manifest, sha256_file
from stc.adapters.inference.onnx_classifier import OnnxClassifier, load_onnx_classifier
from stc.adapters.models.skops_store import SkopsBaselineStore
from stc.adapters.models.tfidf_linear import TfidfLinearTrainer, load_baseline
from stc.adapters.reporting.writers import CompositeReportWriter
from stc.adapters.storage.access_log import JsonlAccessLog
from stc.adapters.storage.fs_model_store import FsModelStore
from stc.application.ports import (
    Classifier,
    ClassifierTrainer,
    DatasetRepository,
    ExperimentTracker,
    HyperparameterSearcher,
    ModelExporter,
    OnnxModelLoader,
    RawDatasetSource,
    ScreeningRunner,
)
from stc.application.services.calibrated_classifier import CalibratedClassifier
from stc.application.services.classifier_factory import ClassifierFactory
from stc.application.services.model_resolver import ModelResolver
from stc.application.services.tracker_factory import TrackerFactory
from stc.application.use_cases.final_report import FinalReportFigureWriter
from stc.application.use_cases.prepare_data import PrepareDataPipeline
from stc.application.use_cases.train_baseline import TrainBaseline
from stc.config import Settings
from stc.domain.calibration import Calibration
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


def build_report_writer() -> CompositeReportWriter:
    """Wire the deterministic JSON/CSV/Markdown report writer for a CLI invocation."""
    return CompositeReportWriter()


def build_access_log(path: Path) -> JsonlAccessLog:
    """Wire the append-only test-access audit log for a CLI invocation."""
    return JsonlAccessLog(path)


def load_classifier(model: str, settings: Settings | None = None) -> Classifier:
    """Resolve a model path, installed name or ``baseline`` to a classifier."""
    resolved = settings if settings is not None else build_settings()
    path = build_model_resolver(resolved).resolve(model)
    return load_classifier_from_path(path, model, resolved)


def artifact_sha256(path: Path) -> str:
    """Return the sha256 of the primary scored artifact in a model directory.

    Bundles expose the ONNX file named by their manifest; a baseline directory
    exposes ``baseline.skops``. The hash is what the audit log records so a test
    read can be tied to the exact weights that produced it.
    """
    root = Path(path)
    if (root / "baseline.skops").is_file():
        return sha256_file(root / "baseline.skops")
    if (root / "manifest.json").is_file():
        manifest = read_manifest(root)
        onnx_section = manifest.get("onnx")
        relative = onnx_section.get("file") if isinstance(onnx_section, Mapping) else None
        declared = relative if isinstance(relative, str) and relative else "model.onnx"
        candidate = root / declared
        if candidate.is_file():
            return sha256_file(candidate)
    return "unknown"


def load_classifier_from_path(
    path: Path,
    model: str,
    settings: Settings | None = None,
) -> Classifier:
    """Build the classifier adapter for an already resolved artifact directory."""
    resolved = settings if settings is not None else build_settings()
    if (path / "baseline.skops").is_file():
        kind = "tfidf-linear"
    elif (path / "manifest.json").is_file():
        kind = str(read_manifest(path)["kind"])
    else:
        raise ArtifactNotFoundError(
            f"model not found: {model}; run: stc model pull {model} "
            "(or stc train baseline for a local baseline)"
        )
    return build_classifier_factory(resolved).create(kind, path)


@dataclass(frozen=True, slots=True)
class CalibratedArtifact:
    """A resolved model plus the calibration stored next to it (D10)."""

    classifier: CalibratedClassifier
    path: Path
    calibration: Calibration | None
    calibration_payload: Mapping[str, object]


def _read_calibration(path: Path) -> tuple[Calibration | None, Mapping[str, object]]:
    calibration_path = path / "calibration.json"
    if not calibration_path.is_file():
        return None, {}
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"calibration file is not valid JSON: {calibration_path}"
        raise InvalidInputError(message) from error
    if not isinstance(payload, Mapping):
        message = f"calibration file must be a JSON object: {calibration_path}"
        raise InvalidInputError(message)
    calibration = Calibration.from_dict(cast(Mapping[str, object], payload))
    return calibration, cast(Mapping[str, object], payload)


def load_calibrated_classifier(
    model: str,
    settings: Settings | None = None,
) -> CalibratedArtifact:
    """Resolve a selector and apply the temperature/review policy stored beside it.

    A model without ``calibration.json`` is served with the identity
    temperature and a review threshold of zero (nothing is flagged), which keeps
    prediction usable before ``stc calibrate`` runs while never inventing a
    threshold.
    """
    resolved = settings if settings is not None else build_settings()
    path = build_model_resolver(resolved).resolve(model)
    inner = load_classifier_from_path(path, model, resolved)
    calibration, payload = _read_calibration(path)
    effective = (
        calibration
        if calibration is not None
        else Calibration(
            temperature=1.0,
            review_threshold=0.0,
        )
    )
    return CalibratedArtifact(
        classifier=CalibratedClassifier(inner, effective),
        path=path,
        calibration=calibration,
        calibration_payload=payload,
    )


def baseline_available(settings: Settings | None = None) -> bool:
    """Return whether the ``baseline`` selector resolves to an artifact."""
    resolved = settings if settings is not None else build_settings()
    try:
        build_model_resolver(resolved).resolve("baseline")
    except ArtifactNotFoundError:
        return False
    return True


def build_classifier_factory(settings: Settings | None = None) -> ClassifierFactory:
    """Register every classifier adapter behind the pure Strategy factory."""
    resolved = settings if settings is not None else build_settings()

    def onnx(path: Path) -> Classifier:
        return OnnxClassifier.from_bundle(path, threads=resolved.threads)

    def baseline(path: Path) -> Classifier:
        return load_baseline(path)

    return ClassifierFactory({"onnx-seqcls": onnx, "tfidf-linear": baseline})


def build_model_store(settings: Settings | None = None) -> FsModelStore:
    """Wire the filesystem model store under ``STC_HOME``."""
    resolved = settings if settings is not None else build_settings()
    return FsModelStore(resolved.resolve_home() / "models", bundle=build_bundle_archive())


def build_model_resolver(settings: Settings | None = None) -> ModelResolver:
    """Resolve CLI model selectors against the store and ``artifacts/``."""
    resolved = settings if settings is not None else build_settings()
    return ModelResolver(
        build_model_store(resolved),
        default_name=resolved.model,
        baseline_dir=Path("artifacts") / "baseline",
    )


def build_bundle_archive() -> BundleArchiveAdapter:
    """Wire the bundle filesystem helper shared by pull, pack and calibrate."""
    return BundleArchiveAdapter()


def build_model_hub(settings: Settings | None = None) -> GithubReleaseHub:
    """Wire the release downloader, honouring ``STC_OFFLINE``."""
    resolved = settings if settings is not None else build_settings()
    return GithubReleaseHub(build_bundle_archive(), offline=resolved.offline)


class _OnnxLoader:
    """Default-thread ONNX loader injected into the export use case."""

    def __init__(self, threads: int) -> None:
        self._threads = threads

    def load(
        self,
        bundle_dir: Path,
        model_path: Path | None = None,
        *,
        threads: int | None = None,
    ) -> Classifier:
        return load_onnx_classifier(
            bundle_dir,
            model_path,
            threads=self._threads if threads is None else threads,
        )


def build_onnx_loader(settings: Settings | None = None) -> OnnxModelLoader:
    """Return a loader that applies the configured ORT thread count."""
    resolved = settings if settings is not None else build_settings()
    return _OnnxLoader(resolved.threads)


def build_registry(path: Path) -> ModelRegistry:
    """Load the committed model registry metadata."""
    return ModelRegistry.from_file(path)


def upsert_registry_entry(
    path: Path,
    *,
    name: str,
    version: str,
    url: str,
    sha256: str,
    size_bytes: int,
    license: str,
) -> None:
    """Insert or replace one release entry in the registry JSON."""
    registry = ModelRegistry.from_file(path) if path.is_file() else ModelRegistry.empty()
    registry.upsert(
        RegistryEntry(
            name=name,
            version=version,
            url=url,
            sha256=sha256,
            size_bytes=size_bytes,
            license=license,
        )
    ).save(path)


def build_exporter() -> ModelExporter:
    """Wire the ONNX exporter lazily, with the shared tiny/HF model builder."""
    module = importlib.import_module("stc.adapters.export.onnx_export")

    def base_loader(backbone: str, n_labels: int, tokenizer: object) -> object:
        factory = importlib.import_module("stc.adapters.training.model_factory")
        return factory.build_base_model(backbone, n_labels, tokenizer=tokenizer)

    exporter = cast(Callable[[Callable[..., object]], ModelExporter], module.OnnxExporter)
    return exporter(base_loader)


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


def build_final_report_figures() -> FinalReportFigureWriter | None:
    """Wire the optional matplotlib figure writer, or ``None`` without the extra."""
    if importlib.util.find_spec("matplotlib") is None:
        return None
    module = importlib.import_module("stc.adapters.reporting.plots")
    factory = cast(Callable[[], FinalReportFigureWriter], module.PlotReportWriter)
    return factory()


def render_results(
    *,
    summary_path: Path,
    table_path: Path,
    final_dir: Path | None = None,
    readme: Path | None = None,
    readme_es: Path | None = None,
) -> tuple[Path, Path]:
    """Wire the T6 results renderer for ``stc report render``."""
    module = importlib.import_module("stc.adapters.reporting.results")
    renderer = cast(Callable[..., tuple[Path, Path]], module.render_results)
    return renderer(
        summary_path=summary_path,
        table_path=table_path,
        final_dir=final_dir,
        readme=readme,
        readme_es=readme_es,
    )


def build_api_app(settings: Settings, *, model_selector: str | None = None) -> FastAPI:
    """Wire the FastAPI adapter without making adaptations depend on each other."""
    app_module = importlib.import_module("stc.adapters.api.app")
    provider_module = importlib.import_module("stc.adapters.api.provider")
    provider = provider_module.BootstrapClassifierProvider(
        settings,
        default_selector=model_selector,
    )
    return cast("FastAPI", app_module.create_app(settings, provider))
