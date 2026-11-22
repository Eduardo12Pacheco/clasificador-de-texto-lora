"""Compose the audited final evaluation report (S6-S10).

``FinalReport`` is one of the only two code paths allowed to read the test
split. It computes every S6-S10 product for the baseline and the shipped model,
writes the report files and appends exactly one access-log entry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from stc.application.dto import ModelInfo
from stc.application.ports import AccessLog, Classifier, DatasetRepository, ReportWriter
from stc.application.services.dev_only_repository import DevOnlyRepository
from stc.application.use_cases.evaluate_model import EvaluateModel, EvaluationResult
from stc.application.use_cases.robustness import (
    RobustnessEvaluator,
    RobustnessReport,
    SliceEvaluator,
)
from stc.domain.access import TestAccessEntry, TestAccessGrant
from stc.domain.calibration import (
    Calibration,
    ReliabilityBins,
    reliability_bins,
    softmax,
)
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.labels import LabelSpace
from stc.domain.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    present_labels,
    top_confusions,
)
from stc.domain.perturbations import PERTURBATIONS
from stc.domain.splits import Split
from stc.domain.stats import (
    ConfidenceInterval,
    McNemarResult,
    bootstrap_ci,
    mcnemar_exact,
    paired_bootstrap_delta,
)

__all__ = [
    "FinalReport",
    "FinalReportFigureWriter",
    "FinalReportResult",
    "ModelEvaluation",
]

BASELINE_KEY = "baseline"
MODEL_KEY = "lora"
VERDICTS = ("transformer_better", "baseline_better", "tie")


class FinalReportFigureWriter(Protocol):
    """Port for the figures the final report renders."""

    def reliability_diagram(self, bins: ReliabilityBins, path: Path) -> None:
        """Render a reliability diagram."""
        ...

    def confusion_matrix(self, matrix: object, intents: Sequence[str], path: Path) -> None:
        """Render a row-normalised confusion matrix."""
        ...

    def per_class_f1(self, intents: Sequence[str], f1_scores: Sequence[float], path: Path) -> None:
        """Render per-intent F1 bars."""
        ...

    def robustness(self, report: RobustnessReport, path: Path) -> None:
        """Render the before/after robustness bars."""
        ...


def _model_info_payload(info: ModelInfo) -> dict[str, object]:
    return {
        "id": info.model_id,
        "kind": info.kind,
        "license": info.license,
        "n_intents": info.n_intents,
        "n_scenarios": info.n_scenarios,
        "parameters": info.parameters,
        "quantization": info.quantization,
        "version": info.version,
    }


def _scores(classifier: Classifier, texts: Sequence[str]) -> NDArray[np.float64]:
    labels = classifier.labels
    scores = np.asarray(classifier.predict_scores(texts), dtype=np.float64)
    if scores.shape != (len(texts), labels.n_intents):
        raise InvalidInputError(
            f"classifier returned scores with shape {scores.shape}, "
            f"expected {(len(texts), labels.n_intents)}"
        )
    if not np.all(np.isfinite(scores)):
        raise InvalidInputError("classifier returned non-finite scores")
    return scores


def _macro_metric(
    n_classes: int, labels: NDArray[np.int64]
) -> Callable[[NDArray[np.int64], NDArray[np.int64]], float]:
    def metric(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
        return macro_f1(confusion_matrix(y_true, y_pred, n_classes), labels)

    return metric


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Predictions and report payload for one model on the test split."""

    key: str
    payload: dict[str, object]
    result: EvaluationResult
    probabilities: NDArray[np.float64]
    correct: NDArray[np.bool_]
    predicted: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class FinalReportResult:
    """The verdict and the files written by one final-report run."""

    verdict: str
    verdict_sentences: Mapping[str, str]
    n_examples: int
    out_dir: Path
    figures_dir: Path
    paths: Mapping[str, Path]


class FinalReport:
    """Compute and persist every audited final-report artifact."""

    def __init__(
        self,
        repository: DatasetRepository,
        writer: ReportWriter,
        access_log: AccessLog,
        *,
        figures: FinalReportFigureWriter | None = None,
    ) -> None:
        self._repository = DevOnlyRepository(repository)
        self._writer = writer
        self._access_log = access_log
        self._figures = figures

    def _evaluate(
        self,
        key: str,
        classifier: Classifier,
        examples: Sequence[Example],
        calibration: Calibration | None,
    ) -> ModelEvaluation:
        result = EvaluateModel().run(
            classifier,
            examples,
            split=Split.DEV,
            calibration=calibration,
        )
        texts = [example.text for example in examples]
        scores = _scores(classifier, texts)
        temperature = calibration.temperature if calibration is not None else 1.0
        probabilities = softmax(scores, T=temperature)
        predicted = np.argmax(probabilities, axis=1).astype(np.int64)
        y_true = np.array(
            [classifier.labels.intent_index(example.intent) for example in examples],
            dtype=np.int64,
        )
        correct = predicted == y_true
        present = present_labels(y_true)
        macro_metric = _macro_metric(classifier.labels.n_intents, present)
        accuracy_ci = bootstrap_ci(accuracy, y_true, predicted)
        macro_ci = bootstrap_ci(macro_metric, y_true, predicted)

        payload = result.to_dict()
        payload["split"] = Split.TEST.value
        payload["model"] = _model_info_payload(classifier.info)
        payload["ci"] = {
            "accuracy": {
                "estimate": accuracy_ci.estimate,
                "lower": accuracy_ci.lower,
                "upper": accuracy_ci.upper,
            },
            "macro_f1": {
                "estimate": macro_ci.estimate,
                "lower": macro_ci.lower,
                "upper": macro_ci.upper,
            },
        }
        payload["top_confusions"] = [
            {
                "count": entry.count,
                "predicted_intent": classifier.labels.intents[entry.predicted],
                "true_intent": classifier.labels.intents[entry.true],
            }
            for entry in top_confusions(result.confusion, 15)
        ]
        return ModelEvaluation(
            key=key,
            payload=payload,
            result=result,
            probabilities=probabilities,
            correct=correct,
            predicted=predicted,
        )

    def run(
        self,
        baseline: Classifier,
        model: Classifier,
        calibrations: Mapping[str, Calibration | None],
        label_space: LabelSpace,
        grant: TestAccessGrant | None,
        out_dir: Path,
        figures_dir: Path,
        *,
        command: str = "stc report final",
        git_sha: str = "unknown",
        robustness_names: Sequence[str] = tuple(PERTURBATIONS),
    ) -> FinalReportResult:
        """Read test once, evaluate both models and write every report file."""
        for key, classifier in ((BASELINE_KEY, baseline), (MODEL_KEY, model)):
            if classifier.labels.intents != label_space.intents:
                raise InvalidInputError(
                    f"{key} classifier label space does not match the supplied label space"
                )
        examples = list(self._repository.load_test(grant))
        if not examples:
            raise InvalidInputError("final evaluation needs at least one example")
        if grant is None:
            raise InvalidInputError("final evaluation needs a test-access grant")
        self._access_log.record(
            TestAccessEntry(
                grant=grant,
                command=command,
                git_sha=git_sha,
                n_examples=len(examples),
            )
        )

        baseline_eval = self._evaluate(
            BASELINE_KEY, baseline, examples, calibrations.get(BASELINE_KEY)
        )
        model_eval = self._evaluate(MODEL_KEY, model, examples, calibrations.get(MODEL_KEY))

        y_true = np.array(
            [model.labels.intent_index(example.intent) for example in examples], dtype=np.int64
        )
        present = present_labels(y_true)
        macro_metric = _macro_metric(model.labels.n_intents, present)
        delta_accuracy = paired_bootstrap_delta(
            accuracy, y_true, model_eval.predicted, baseline_eval.predicted
        )
        delta_macro = paired_bootstrap_delta(
            macro_metric, y_true, model_eval.predicted, baseline_eval.predicted
        )
        mcnemar = mcnemar_exact(model_eval.correct, baseline_eval.correct)
        verdict = _verdict(delta_macro.lower, delta_macro.upper, mcnemar.p)
        sentences = {
            "en": _verdict_sentence("en", verdict, delta_macro, mcnemar),
            "es": _verdict_sentence("es", verdict, delta_macro, mcnemar),
        }

        robustness = RobustnessEvaluator()
        slices = SliceEvaluator()
        robustness_reports = {
            BASELINE_KEY: robustness.run(
                baseline,
                examples,
                robustness_names,
                calibrations.get(BASELINE_KEY),
                split=Split.TEST,
            ),
            MODEL_KEY: robustness.run(
                model,
                examples,
                robustness_names,
                calibrations.get(MODEL_KEY),
                split=Split.TEST,
            ),
        }
        slice_reports = {
            BASELINE_KEY: slices.run(
                baseline,
                examples,
                calibrations.get(BASELINE_KEY),
                split=Split.TEST,
            ),
            MODEL_KEY: slices.run(
                model,
                examples,
                calibrations.get(MODEL_KEY),
                split=Split.TEST,
            ),
        }

        out_dir = Path(out_dir)
        figures_dir = Path(figures_dir)
        paths: dict[str, Path] = {}
        paths["metrics"] = out_dir / "metrics.json"
        self._writer.write_json(
            paths["metrics"],
            {
                "schema_version": 1,
                "split": Split.TEST.value,
                "n_examples": len(examples),
                "models": {
                    BASELINE_KEY: baseline_eval.payload,
                    MODEL_KEY: model_eval.payload,
                },
            },
        )
        paths["comparison"] = out_dir / "comparison.json"
        self._writer.write_json(
            paths["comparison"],
            {
                "schema_version": 1,
                "split": Split.TEST.value,
                "n_examples": len(examples),
                "models": [BASELINE_KEY, MODEL_KEY],
                "delta": {
                    "accuracy": _ci_payload(delta_accuracy),
                    "macro_f1": _ci_payload(delta_macro),
                },
                "mcnemar": {"b": mcnemar.b, "c": mcnemar.c, "p": mcnemar.p},
                "verdict": verdict,
                "verdict_sentences": sentences,
            },
        )
        paths["calibration_test"] = out_dir / "calibration_test.json"
        self._writer.write_json(
            paths["calibration_test"],
            {
                "schema_version": 1,
                "split": Split.TEST.value,
                "n_examples": len(examples),
                "models": {
                    BASELINE_KEY: _calibration_payload(baseline_eval.result),
                    MODEL_KEY: _calibration_payload(model_eval.result),
                },
            },
        )
        paths["robustness"] = out_dir / "robustness.json"
        self._writer.write_json(
            paths["robustness"],
            {
                "schema_version": 1,
                "split": Split.TEST.value,
                "n_examples": len(examples),
                "models": {
                    BASELINE_KEY: robustness_reports[BASELINE_KEY].to_dict(),
                    MODEL_KEY: robustness_reports[MODEL_KEY].to_dict(),
                },
            },
        )
        paths["slices"] = out_dir / "slices.json"
        self._writer.write_json(
            paths["slices"],
            {
                "schema_version": 1,
                "split": Split.TEST.value,
                "n_examples": len(examples),
                "models": {
                    BASELINE_KEY: slice_reports[BASELINE_KEY].to_dict(),
                    MODEL_KEY: slice_reports[MODEL_KEY].to_dict(),
                },
            },
        )
        paths["per_class"] = out_dir / "per_class.csv"
        self._writer.write_csv(
            paths["per_class"],
            ["model", "intent", "precision", "recall", "f1", "support"],
            _per_class_rows(baseline_eval, model_eval),
        )
        paths["confusion_matrix"] = out_dir / "confusion_matrix.csv"
        self._writer.write_csv(
            paths["confusion_matrix"],
            ["model", "true_intent", "predicted_intent", "count"],
            [
                [
                    evaluation.key,
                    evaluation.result.intents[true],
                    evaluation.result.intents[pred],
                    int(evaluation.result.confusion[true, pred]),
                ]
                for evaluation in (baseline_eval, model_eval)
                for true in range(len(evaluation.result.intents))
                for pred in range(len(evaluation.result.intents))
            ],
        )
        paths["errors"] = out_dir / "errors.json"
        self._writer.write_json(
            paths["errors"],
            _errors_payload(examples, baseline_eval, model_eval),
        )

        if self._figures is not None:
            self._write_figures(figures_dir, baseline_eval, model_eval, robustness_reports)
            paths["figures"] = figures_dir

        return FinalReportResult(
            verdict=verdict,
            verdict_sentences=sentences,
            n_examples=len(examples),
            out_dir=out_dir,
            figures_dir=figures_dir,
            paths=paths,
        )

    def _write_figures(
        self,
        figures_dir: Path,
        baseline_eval: ModelEvaluation,
        model_eval: ModelEvaluation,
        robustness_reports: Mapping[str, RobustnessReport],
    ) -> None:
        figures = self._figures
        if figures is None:
            return
        for evaluation in (baseline_eval, model_eval):
            bins = reliability_bins(evaluation.probabilities.max(axis=1), evaluation.correct)
            figures.reliability_diagram(bins, figures_dir / f"reliability_{evaluation.key}.png")
            figures.confusion_matrix(
                evaluation.result.confusion,
                evaluation.result.intents,
                figures_dir / f"confusion_{evaluation.key}.png",
            )
        figures.per_class_f1(
            list(model_eval.result.intents),
            [row.f1 for row in model_eval.result.per_class],
            figures_dir / "per_class_f1.png",
        )
        figures.robustness(robustness_reports[MODEL_KEY], figures_dir / "robustness.png")


def _ci_payload(interval: ConfidenceInterval) -> dict[str, float]:
    return {
        "estimate": interval.estimate,
        "lower": interval.lower,
        "upper": interval.upper,
    }


def _calibration_payload(result: EvaluationResult) -> dict[str, object]:
    metrics = result.metrics
    calibrated = metrics.calibrated
    return {
        "temperature": metrics.temperature,
        "review_threshold": metrics.review_threshold,
        "before": {
            "ece": metrics.uncalibrated_ece,
            "nll": metrics.uncalibrated_nll,
            "brier": metrics.uncalibrated_brier,
        },
        "after": (
            {
                "ece": metrics.calibrated_ece,
                "nll": metrics.calibrated_nll,
                "brier": metrics.calibrated_brier,
            }
            if calibrated
            else None
        ),
        "coverage_at_threshold": metrics.coverage_at_threshold,
        "selective_accuracy": metrics.selective_accuracy,
    }


def _per_class_rows(
    baseline_eval: ModelEvaluation, model_eval: ModelEvaluation
) -> list[list[object]]:
    output: list[list[object]] = []
    for evaluation in (baseline_eval, model_eval):
        for row in evaluation.result.per_class:
            output.append(
                [
                    evaluation.key,
                    row.intent,
                    row.precision,
                    row.recall,
                    row.f1,
                    row.support,
                ]
            )
    return output


def _errors_payload(
    examples: Sequence[Example],
    baseline_eval: ModelEvaluation,
    model_eval: ModelEvaluation,
) -> dict[str, object]:
    intents = model_eval.result.intents
    y_true = np.array(
        [model_eval.result.intents.index(example.intent) for example in examples], dtype=np.int64
    )
    baseline_confidence = baseline_eval.probabilities.max(axis=1)
    model_confidence = model_eval.probabilities.max(axis=1)
    rows: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        if baseline_eval.correct[index] and model_eval.correct[index]:
            continue
        rows.append(
            {
                "example_id": example.id,
                "text": example.text,
                "true_intent": intents[int(y_true[index])],
                BASELINE_KEY: {
                    "confidence": float(baseline_confidence[index]),
                    "correct": bool(baseline_eval.correct[index]),
                    "predicted_intent": intents[int(baseline_eval.predicted[index])],
                },
                MODEL_KEY: {
                    "confidence": float(model_confidence[index]),
                    "correct": bool(model_eval.correct[index]),
                    "predicted_intent": intents[int(model_eval.predicted[index])],
                },
            }
        )
    return {
        "schema_version": 1,
        "split": Split.TEST.value,
        "n_examples": len(examples),
        "n_errors": len(rows),
        "errors": rows,
    }


def _verdict(macro_lower: float, macro_upper: float, p_value: float) -> str:
    if macro_lower > 0.0 and p_value < 0.05:
        return "transformer_better"
    if macro_upper < 0.0 and p_value < 0.05:
        return "baseline_better"
    return "tie"


def _verdict_sentence(
    language: str,
    verdict: str,
    delta: ConfidenceInterval,
    mcnemar: McNemarResult,
) -> str:
    macro = f"Δmacro-F1 {delta.estimate:+.4f} (95 % CI [{delta.lower:+.4f}, {delta.upper:+.4f}])"
    stats = f"{macro}, McNemar p = {mcnemar.p:.3f}"
    # A significant McNemar test with a macro-F1 tie means the paired accuracy
    # difference is real even though macro-F1 is not. ``b`` counts examples only
    # the LoRA transformer got right, ``c`` the reverse.
    significant_accuracy = mcnemar.p < 0.05
    lora_more_accurate = mcnemar.b > mcnemar.c
    if language == "es":
        if verdict == "transformer_better":
            return (
                "El transformer con LoRA supera de forma significativa a la línea base "
                f"TF-IDF en el test auditado ({stats})."
            )
        if verdict == "baseline_better":
            return (
                "La línea base TF-IDF supera de forma significativa al transformer con LoRA "
                f"en el test auditado ({stats})."
            )
        if significant_accuracy:
            winner = (
                "el transformer con LoRA es más exacto que la línea base TF-IDF"
                if lora_more_accurate
                else "la línea base TF-IDF es más exacta que el transformer con LoRA"
            )
            return (
                f"La diferencia de exactitud es estadísticamente significativa: {winner} "
                f"en el test auditado (McNemar p = {mcnemar.p:.3f}); la diferencia de "
                f"macro-F1 no es significativa ({macro})."
            )
        return (
            "El transformer con LoRA y la línea base TF-IDF rinden dentro del ruido "
            f"estadístico en el test auditado ({stats}); el resultado es un empate."
        )
    if verdict == "transformer_better":
        return (
            "The LoRA transformer significantly outperforms the TF-IDF baseline "
            f"on the audited test split ({stats})."
        )
    if verdict == "baseline_better":
        return (
            "The TF-IDF baseline significantly outperforms the LoRA transformer "
            f"on the audited test split ({stats})."
        )
    if significant_accuracy:
        winner = (
            "the LoRA transformer is more accurate than the TF-IDF baseline"
            if lora_more_accurate
            else "the TF-IDF baseline is more accurate than the LoRA transformer"
        )
        return (
            f"The difference in accuracy is statistically significant: {winner} "
            f"on the audited test split (McNemar p = {mcnemar.p:.3f}); the macro-F1 "
            f"difference is not significant ({macro})."
        )
    return (
        "The LoRA transformer and the TF-IDF baseline perform within statistical noise "
        f"on the audited test split ({stats}); the outcome is a tie."
    )
