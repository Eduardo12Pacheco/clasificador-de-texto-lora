"""Baseline grid parsing, winner selection and the train-baseline use case."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from stc.application.dto import (
    BaselineCandidate,
    BaselineGrid,
    DatasetMeta,
    ModelInfo,
)
from stc.application.use_cases.train_baseline import TrainBaseline
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace
from stc.domain.splits import Split

pytestmark = pytest.mark.unit

INTENTS = ("alarm_cancel", "alarm_set", "play_music", "weather_query")


def candidate(
    *,
    macro_f1: float,
    accuracy: float,
    n_features: int = 100,
    label: str = "x",
) -> BaselineCandidate:
    return BaselineCandidate(
        params={"label": label},
        dev_accuracy=accuracy,
        dev_macro_f1=macro_f1,
        n_features=n_features,
        fit_seconds=0.1,
    )


class FakeRepository:
    def __init__(self, train: Sequence[Example], dev: Sequence[Example]) -> None:
        self._splits = {Split.TRAIN: tuple(train), Split.DEV: tuple(dev)}

    def load(self, split: Split) -> Sequence[Example]:
        return self._splits[split]

    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            dataset_id="fake",
            version="1.0",
            tarball_sha256="a" * 64,
            stc_version="0.1.0",
            raw_counts={"train": len(self._splits[Split.TRAIN]), "dev": 2, "test": 0},
            processed_counts={"train": len(self._splits[Split.TRAIN]), "dev": 2, "test": 0},
            dropped_counts={},
        )


class FakeClassifier:
    def __init__(self, labels: LabelSpace) -> None:
        self._labels = labels

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            model_id="fake-baseline",
            kind="tfidf-linear",
            version="0.1.0",
            quantization=None,
            parameters=10,
            license="MIT",
            n_intents=self._labels.n_intents,
            n_scenarios=self._labels.n_scenarios,
        )

    @property
    def labels(self) -> LabelSpace:
        return self._labels

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts), self._labels.n_intents), dtype=np.float32)


class FakeTrainer:
    def __init__(self, candidates: Sequence[BaselineCandidate]) -> None:
        self.candidates = tuple(candidates)
        self.fitted: list[Mapping[str, object]] = []
        self.quick: bool | None = None

    def evaluate_configs(
        self,
        train: Sequence[Example],
        dev: Sequence[Example],
        labels: LabelSpace,
        grid: BaselineGrid,
        *,
        quick: bool = False,
    ) -> Sequence[BaselineCandidate]:
        self.quick = quick
        return self.candidates

    def fit(
        self,
        train: Sequence[Example],
        labels: LabelSpace,
        params: Mapping[str, object],
    ) -> FakeClassifier:
        self.fitted.append(params)
        return FakeClassifier(labels)


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[tuple[FakeClassifier, Path, Mapping[str, object]]] = []

    def save(
        self,
        classifier: FakeClassifier,
        dest: Path,
        *,
        metadata: Mapping[str, object],
    ) -> Mapping[str, object]:
        self.saved.append((classifier, dest, metadata))
        return {"kind": "tfidf-linear", "winner": dict(metadata["winner"])}

    def load(self, source: Path) -> FakeClassifier:
        raise AssertionError("the train use case never loads a baseline")


class FakeWriter:
    def __init__(self) -> None:
        self.json: dict[Path, Mapping[str, object]] = {}

    def write_json(self, path: Path, payload: Mapping[str, object]) -> None:
        self.json[path] = payload

    def write_csv(
        self,
        path: Path,
        header: Sequence[str],
        rows: Sequence[Sequence[object]],
    ) -> None:
        raise AssertionError("unused")

    def write_markdown(self, path: Path, text: str) -> None:
        raise AssertionError("unused")

    def write_figure(self, path: Path, figure: object) -> None:
        raise AssertionError("unused")


def examples(intent: str, count: int = 2) -> list[Example]:
    return [
        Example(
            id=str(index),
            text=f"texto {intent} {index}",
            intent=intent,
            scenario=intent.split("_", maxsplit=1)[0],
            worker_id="w1",
        )
        for index in range(count)
    ]


def test_use_case_writes_reports_and_persists_the_winner(tmp_path: Path) -> None:
    train = [row for intent in INTENTS for row in examples(intent)]
    dev = [row for intent in INTENTS for row in examples(intent, 1)]
    candidates = [
        candidate(macro_f1=0.6, accuracy=0.7, label="loser"),
        candidate(macro_f1=0.9, accuracy=0.8, label="winner"),
    ]
    trainer = FakeTrainer(candidates)
    store = FakeStore()
    writer = FakeWriter()
    reports = tmp_path / "reports"

    result = TrainBaseline(trainer, store, writer).run(
        FakeRepository(train, dev),
        out_dir=tmp_path / "artifacts",
        grid=BaselineGrid.default(),
        reports_dir=reports,
    )

    assert result.winner.params["label"] == "winner"
    assert result.n_configs == 2
    assert trainer.fitted == [{"label": "winner"}]
    assert result.search_path == reports / "search.json"
    assert result.dev_metrics_path == reports / "dev_metrics.json"
    search = writer.json[result.search_path]
    assert search["n_configs"] == 2
    assert len(search["configs"]) == 2
    assert search["selection"]["winner"] == {"label": "winner"}
    dev_metrics = writer.json[result.dev_metrics_path]
    assert dev_metrics["winner"] == {"label": "winner"}
    assert dev_metrics["split"] == "dev"
    _classifier, dest, metadata = store.saved[0]
    assert dest == tmp_path / "artifacts"
    assert metadata["winner"] == {"label": "winner"}
    assert metadata["dataset"]["dataset_id"] == "fake"


def test_use_case_defaults_reports_to_the_artifact_directory(tmp_path: Path) -> None:
    train = [row for intent in INTENTS for row in examples(intent)]
    dev = [row for intent in INTENTS for row in examples(intent, 1)]
    writer = FakeWriter()
    use_case = TrainBaseline(
        FakeTrainer([candidate(macro_f1=0.9, accuracy=0.9)]),
        FakeStore(),
        writer,
    )

    result = use_case.run(
        FakeRepository(train, dev),
        out_dir=tmp_path / "artifacts",
        grid=BaselineGrid.default(),
    )

    assert result.search_path == tmp_path / "artifacts" / "search.json"
    assert result.dev_metrics_path == tmp_path / "artifacts" / "dev_metrics.json"


def test_use_case_forwards_quick_to_the_trainer(tmp_path: Path) -> None:
    trainer = FakeTrainer([candidate(macro_f1=0.3, accuracy=0.4)])
    train = [row for intent in INTENTS for row in examples(intent)]
    dev = [row for intent in INTENTS for row in examples(intent, 1)]

    TrainBaseline(trainer, FakeStore(), FakeWriter()).run(
        FakeRepository(train, dev),
        out_dir=tmp_path,
        grid=BaselineGrid.default(),
        quick=True,
    )

    assert trainer.quick is True


def test_use_case_rejects_an_empty_candidate_list(tmp_path: Path) -> None:
    from stc.domain.errors import InvalidInputError

    with pytest.raises(InvalidInputError):
        TrainBaseline(FakeTrainer([]), FakeStore(), FakeWriter()).run(
            FakeRepository(
                [row for intent in INTENTS for row in examples(intent)],
                [row for intent in INTENTS for row in examples(intent, 1)],
            ),
            out_dir=tmp_path,
            grid=BaselineGrid.default(),
        )


def test_search_report_is_json_serialisable_and_deterministic(tmp_path: Path) -> None:
    train = [row for intent in INTENTS for row in examples(intent)]
    dev = [row for intent in INTENTS for row in examples(intent, 1)]
    writer_one = FakeWriter()
    writer_two = FakeWriter()
    candidates = [candidate(macro_f1=0.5, accuracy=0.6, label="a")]

    for writer in (writer_one, writer_two):
        TrainBaseline(FakeTrainer(candidates), FakeStore(), writer).run(
            FakeRepository(train, dev),
            out_dir=tmp_path,
            grid=BaselineGrid.default(),
        )

    text_one = json.dumps(writer_one.json[tmp_path / "search.json"], sort_keys=True)
    text_two = json.dumps(writer_two.json[tmp_path / "search.json"], sort_keys=True)
    assert text_one == text_two
