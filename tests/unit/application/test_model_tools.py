"""Calibration, benchmark, prediction, resolver and factory use cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from stc.application.dto import ModelInfo
from stc.application.services.calibrated_classifier import CalibratedClassifier
from stc.application.services.classifier_factory import ClassifierFactory
from stc.application.services.model_resolver import ModelResolver
from stc.application.use_cases.benchmark_model import BenchmarkModel
from stc.application.use_cases.calibrate_model import CalibrateModel
from stc.application.use_cases.predict_texts import PredictTexts
from stc.domain.calibration import Calibration
from stc.domain.errors import ArtifactNotFoundError, InvalidInputError
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace

LABELS = LabelSpace.from_intents(["alarm_set", "music_play", "weather_query"])


class FakeClassifier:
    """Deterministic classifier whose logits are a fixed function of the text."""

    def __init__(self, scale: float = 2.0) -> None:
        self._scale = scale
        self.calls = 0

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            model_id="fake",
            kind="fake",
            version="0.0.1",
            quantization=None,
            parameters=3,
            license="MIT",
            n_intents=LABELS.n_intents,
            n_scenarios=LABELS.n_scenarios,
        )

    @property
    def labels(self) -> LabelSpace:
        return LABELS

    def predict_scores(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls += 1
        rows = []
        for index, text in enumerate(texts):
            base = np.zeros(3, dtype=np.float32)
            base[(index + len(text)) % 3] = 4.0 * self._scale
            base[(index + len(text) + 1) % 3] = 2.0 * self._scale
            rows.append(base)
        return np.asarray(rows, dtype=np.float32)


def _examples(n: int = 120) -> list[Example]:
    intents = LABELS.intents
    return [
        Example(
            id=str(index),
            text=f"utterance number {index}",
            intent=intents[index % len(intents)],
            scenario=intents[index % len(intents)].split("_")[0],
            worker_id="w1",
        )
        for index in range(n)
    ]


def test_calibration_payload_has_the_expected_fields() -> None:
    report = CalibrateModel().run(FakeClassifier(), _examples())
    payload = report.to_dict()
    for key in (
        "temperature",
        "review_threshold",
        "target_selective_accuracy",
        "dev_coverage_at_threshold",
        "ece_dev_before",
        "ece_dev_after",
        "nll_dev_before",
        "nll_dev_after",
    ):
        assert key in payload
    assert payload["temperature"] > 0
    assert 0.0 <= payload["review_threshold"] <= 1.0
    assert payload["n_dev"] == 120


def test_calibration_needs_examples() -> None:
    with pytest.raises(InvalidInputError):
        CalibrateModel().run(FakeClassifier(), [])


def test_benchmark_reports_every_thread_count() -> None:
    counter = {"value": 0.0}

    def clock() -> float:
        counter["value"] += 0.001
        return counter["value"]

    report = BenchmarkModel(clock=clock).run(
        lambda threads: FakeClassifier(),
        [f"text {index}" for index in range(40)],
        threads=[1, 4],
        runs=5,
        environment={"cpu_model": "Test CPU", "onnxruntime_version": "1.0"},
    )
    assert report["cpu_model"] == "Test CPU"
    threads = report["threads"]
    assert isinstance(threads, list)
    assert [row["threads"] for row in threads] == [1, 4]
    for row in threads:
        assert row["p50_ms"] >= 0
        assert row["p95_ms"] >= row["p50_ms"]
        assert row["batch32_throughput_per_s"] > 0


def test_benchmark_rejects_invalid_inputs() -> None:
    with pytest.raises(InvalidInputError):
        BenchmarkModel().run(
            lambda threads: FakeClassifier(), [], threads=[1], runs=1, environment={}
        )
    with pytest.raises(InvalidInputError):
        BenchmarkModel().run(
            lambda threads: FakeClassifier(), ["x"], threads=[], runs=1, environment={}
        )


def test_predict_payload_schema() -> None:
    calibrated = CalibratedClassifier(FakeClassifier(), Calibration(1.5, 0.5))
    payload = PredictTexts().run(calibrated, ["hola", "ponme música"], top_k=2)
    assert payload["schema_version"] == 1
    assert payload["model"]["temperature"] == 1.5
    predictions = payload["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    first = predictions[0]
    assert set(first) >= {"index", "intent", "scenario", "confidence", "needs_review", "top_k"}
    assert len(first["top_k"]) == 2


def test_predict_rejects_blank_texts() -> None:
    calibrated = CalibratedClassifier(FakeClassifier(), Calibration(1.5, 0.5))
    with pytest.raises(InvalidInputError):
        PredictTexts().run(calibrated, ["   "])


def test_classifier_factory_dispatches_by_kind(tmp_path: Path) -> None:
    fake = FakeClassifier()
    factory = ClassifierFactory({"fake": lambda path: fake})
    assert factory.create("fake", tmp_path) is fake
    assert factory.kinds == ("fake",)
    with pytest.raises(InvalidInputError, match="unsupported model kind"):
        factory.create("nope", tmp_path)


class FakeStore:
    def __init__(self, root: Path, installed: Mapping[str, str]) -> None:
        self.root = root
        self._installed = installed

    def save(self, bundle_dir: Path, name: str, version: str) -> Path:
        raise NotImplementedError

    def load(self, name: str, version: str) -> Path:
        key = f"{name}/{version}"
        if key not in self._installed:
            raise ArtifactNotFoundError(f"model {name} {version} is not installed")
        return self.root / key

    def list_installed(self) -> Sequence[ModelInfo]:
        return [
            ModelInfo(
                model_id=name,
                kind="fake",
                version=version,
                quantization=None,
                parameters=1,
                license="MIT",
                n_intents=3,
                n_scenarios=2,
            )
            for name, version in (key.split("/") for key in self._installed)
        ]

    def verify(self, bundle_dir: Path) -> None:
        raise NotImplementedError


def test_resolver_accepts_paths_baseline_and_installed_names(tmp_path: Path) -> None:
    bundle = tmp_path / "models" / "stc-es-intent" / "0.2.0"
    bundle.mkdir(parents=True)
    (bundle / "baseline").mkdir()
    store = FakeStore(tmp_path / "models", {"stc-es-intent/0.1.0": "x", "stc-es-intent/0.2.0": "x"})
    resolver = ModelResolver(store, default_name="stc-es-intent")
    assert resolver.resolve(str(bundle)) == bundle
    assert resolver.resolve("stc-es-intent") == bundle
    assert resolver.resolve("baseline") == bundle / "baseline"


def test_resolver_falls_back_to_the_training_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "artifacts" / "baseline"
    baseline.mkdir(parents=True)
    store = FakeStore(tmp_path / "models", {})
    resolver = ModelResolver(store, default_name="stc-es-intent", baseline_dir=baseline)
    assert resolver.resolve("baseline") == baseline


def test_resolver_explains_a_missing_model(tmp_path: Path) -> None:
    resolver = ModelResolver(FakeStore(tmp_path / "models", {}), default_name="stc-es-intent")
    with pytest.raises(ArtifactNotFoundError, match="stc model pull"):
        resolver.resolve("stc-es-intent")
    with pytest.raises(ArtifactNotFoundError, match="stc train baseline"):
        resolver.resolve("baseline")
