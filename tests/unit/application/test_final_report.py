"""The final-report composer, verdict rule and audit trail (S6-S10)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from stc.adapters.reporting.csv_writer import CsvReportWriter
from stc.adapters.reporting.json_writer import JsonReportWriter
from stc.adapters.reporting.markdown import MarkdownReportWriter
from stc.application.dto import DatasetMeta, ModelInfo
from stc.application.ports import DatasetRepository
from stc.application.use_cases.final_report import FinalReport, _verdict_sentence
from stc.domain.access import TestAccessEntry as AccessEntry
from stc.domain.access import TestAccessGrant as AccessGrant
from stc.domain.calibration import Calibration, ReliabilityBins
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace
from stc.domain.splits import Split
from stc.domain.stats import ConfidenceInterval, McNemarResult

pytestmark = pytest.mark.unit

INTENTS = ("alarm_cancel", "alarm_set", "play_music", "weather_query")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _examples(n: int = 10) -> list[Example]:
    return [
        Example(
            id=str(index + 1),
            text=f"utterance {index}",
            intent=INTENTS[index % len(INTENTS)],
            scenario=INTENTS[index % len(INTENTS)].split("_")[0],
            worker_id="w1",
        )
        for index in range(n)
    ]


class PredictionClassifier:
    def __init__(self, predictions: Sequence[int], *, model_id: str) -> None:
        self._predictions = list(predictions)
        self._model_id = model_id

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self._model_id,
            kind="fake",
            version="0.0.1",
            quantization=None,
            parameters=10,
            license="MIT",
            n_intents=len(INTENTS),
            n_scenarios=4,
        )

    @property
    def labels(self) -> LabelSpace:
        return LabelSpace.from_intents(INTENTS)

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) != len(self._predictions):
            return np.zeros((len(texts), len(INTENTS)), dtype=np.float32)
        scores = np.zeros((len(texts), len(INTENTS)), dtype=np.float32)
        for row, predicted in enumerate(self._predictions):
            scores[row, predicted] = 5.0
        return scores


class FakeRepository(DatasetRepository):
    def __init__(self, examples: Sequence[Example]) -> None:
        self._examples = list(examples)

    def load(self, split: Split) -> Sequence[Example]:
        return tuple(self._examples)

    def save(self, split: Split, examples: Sequence[Example]) -> None:
        raise AssertionError("final report must not write splits")

    def meta(self) -> DatasetMeta:
        raise AssertionError("final report must not need dataset metadata")

    def save_meta(self, meta: DatasetMeta) -> None:
        raise AssertionError("final report must not write metadata")


class FileWriter:
    def write_json(self, path: Path, payload: object) -> None:
        JsonReportWriter().write_json(path, payload)  # type: ignore[arg-type]

    def write_csv(
        self, path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]
    ) -> None:
        CsvReportWriter().write_csv(path, header, rows)

    def write_markdown(self, path: Path, text: str) -> None:
        MarkdownReportWriter().write_markdown(path, text)

    def write_figure(self, path: Path, figure: object) -> None:
        raise AssertionError("figures are written through the figure port")


class RecordingAccessLog:
    def __init__(self) -> None:
        self.entries: list[AccessEntry] = []

    def record(self, entry: AccessEntry) -> None:
        self.entries.append(entry)


class RecordingFigures:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        self.paths.append(path)

    def reliability_diagram(self, bins: ReliabilityBins, path: Path) -> None:
        self._touch(path)

    def confusion_matrix(self, matrix: object, intents: Sequence[str], path: Path) -> None:
        self._touch(path)

    def per_class_f1(self, intents: Sequence[str], f1_scores: Sequence[float], path: Path) -> None:
        self._touch(path)

    def robustness(self, report: object, path: Path) -> None:
        self._touch(path)


def _grant() -> AccessGrant:
    return AccessGrant("final evaluation", "lora", datetime(2026, 11, 20, tzinfo=UTC))


def _run(
    tmp_path: Path,
    model_predictions: Sequence[int],
    baseline_predictions: Sequence[int],
    *,
    figures: RecordingFigures | None = None,
) -> tuple[object, RecordingAccessLog]:
    examples = _examples()
    access_log = RecordingAccessLog()
    report = FinalReport(
        FakeRepository(examples),
        FileWriter(),
        access_log,
        figures=figures,
    )
    result = report.run(
        PredictionClassifier(baseline_predictions, model_id="baseline"),
        PredictionClassifier(model_predictions, model_id="lora"),
        {
            "baseline": Calibration(temperature=1.0, review_threshold=0.5),
            "lora": Calibration(temperature=2.0, review_threshold=0.5),
        },
        LabelSpace.from_intents(INTENTS),
        _grant(),
        tmp_path / "final",
        tmp_path / "figures",
        command="stc report final",
        git_sha="abc1234",
        robustness_names=("uppercase",),
    )
    return result, access_log


def _correct() -> list[int]:
    return [index % len(INTENTS) for index in range(10)]


def test_transformer_better_verdict(tmp_path: Path) -> None:
    result, access_log = _run(tmp_path, _correct(), [0, 1, 2, 3, 1, 2, 3, 0, 1, 2])

    assert result.verdict == "transformer_better"
    assert len(access_log.entries) == 1
    assert access_log.entries[0].n_examples == 10
    assert access_log.entries[0].command == "stc report final"


def test_baseline_better_verdict(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, [0, 1, 2, 3, 1, 2, 3, 0, 1, 2], _correct())

    assert result.verdict == "baseline_better"


def test_tie_verdict(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, _correct(), [1, 2, 2, 3, 0, 1, 2, 3, 0, 1])

    assert result.verdict == "tie"


_MACRO_DELTA = ConfidenceInterval(estimate=0.0131, lower=-0.0084, upper=0.0344)


@pytest.mark.parametrize("language", ["en", "es"])
def test_significant_mcnemar_is_never_described_as_statistical_noise(language: str) -> None:
    sentence = _verdict_sentence(language, "tie", _MACRO_DELTA, McNemarResult(b=206, c=130, p=1e-4))

    assert "noise" not in sentence
    assert "ruido" not in sentence


@pytest.mark.parametrize(
    ("language", "b", "c", "expected"),
    [
        (
            "en",
            206,
            130,
            "the LoRA transformer is more accurate than the TF-IDF baseline",
        ),
        (
            "en",
            130,
            206,
            "the TF-IDF baseline is more accurate than the LoRA transformer",
        ),
        (
            "es",
            206,
            130,
            "el transformer con LoRA es más exacto que la línea base TF-IDF",
        ),
        (
            "es",
            130,
            206,
            "la línea base TF-IDF es más exacta que el transformer con LoRA",
        ),
    ],
)
def test_significant_mcnemar_states_significance_and_the_winner(
    language: str, b: int, c: int, expected: str
) -> None:
    sentence = _verdict_sentence(language, "tie", _MACRO_DELTA, McNemarResult(b=b, c=c, p=1e-4))

    assert expected in sentence
    assert "macro-F1" in sentence


@pytest.mark.parametrize(
    ("language", "phrase"),
    [("en", "statistical noise"), ("es", "ruido estadístico")],
)
def test_non_significant_tie_keeps_the_noise_wording(language: str, phrase: str) -> None:
    sentence = _verdict_sentence(language, "tie", _MACRO_DELTA, McNemarResult(b=2, c=0, p=0.5))

    assert phrase in sentence


def test_final_report_writes_every_output_and_one_audit_line(tmp_path: Path) -> None:
    figures = RecordingFigures()
    result, access_log = _run(
        tmp_path,
        _correct(),
        [0, 1, 2, 3, 1, 2, 3, 0, 1, 2],
        figures=figures,
    )
    out = tmp_path / "final"

    for name in (
        "metrics.json",
        "comparison.json",
        "per_class.csv",
        "confusion_matrix.csv",
        "calibration_test.json",
        "robustness.json",
        "slices.json",
        "errors.json",
    ):
        assert (out / name).is_file(), name
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["models"]) == {"baseline", "lora"}
    assert metrics["split"] == "test"
    assert (
        metrics["models"]["lora"]["ci"]["macro_f1"]["lower"]
        <= (metrics["models"]["lora"]["ci"]["macro_f1"]["estimate"])
    )
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["verdict"] == "transformer_better"
    assert set(comparison["delta"]) == {"accuracy", "macro_f1"}
    assert set(comparison["verdict_sentences"]) == {"en", "es"}
    errors = json.loads((out / "errors.json").read_text(encoding="utf-8"))
    assert len(errors["errors"]) == 6
    with (out / "per_class.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert set(rows[0]) == {"model", "intent", "precision", "recall", "f1", "support"}
    assert len(access_log.entries) == 1
    assert result.verdict in {"transformer_better", "baseline_better", "tie"}
    assert len(figures.paths) == 6
    assert {path.name for path in figures.paths} == {
        "reliability_baseline.png",
        "reliability_lora.png",
        "confusion_baseline.png",
        "confusion_lora.png",
        "per_class_f1.png",
        "robustness.png",
    }


def test_generated_reports_pass_the_final_validator(tmp_path: Path) -> None:
    _run(tmp_path, _correct(), [0, 1, 2, 3, 1, 2, 3, 0, 1, 2])

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_reports.py"),
            "final",
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "OK" in result.stdout


def test_empty_test_split_is_rejected(tmp_path: Path) -> None:
    access_log = RecordingAccessLog()
    report = FinalReport(FakeRepository([]), FileWriter(), access_log)

    with pytest.raises(Exception, match="at least one example"):
        report.run(
            PredictionClassifier([], model_id="baseline"),
            PredictionClassifier([], model_id="lora"),
            {},
            LabelSpace.from_intents(INTENTS),
            _grant(),
            tmp_path / "final",
            tmp_path / "figures",
            robustness_names=("uppercase",),
        )
