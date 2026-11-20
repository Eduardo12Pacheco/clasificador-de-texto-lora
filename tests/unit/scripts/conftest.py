"""Synthetic report trees shared by the report-tooling unit tests."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric_block(key: str, macro_f1: float, accuracy: float) -> dict[str, object]:
    return {
        "n_examples": 10,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": macro_f1,
        "top3_accuracy": 0.98,
        "scenario_accuracy": 0.95,
        "ci": {
            "accuracy": {"estimate": accuracy, "lower": accuracy - 0.02, "upper": accuracy + 0.02},
            "macro_f1": {
                "estimate": macro_f1,
                "lower": macro_f1 - 0.02,
                "upper": macro_f1 + 0.02,
            },
        },
        "model": {"id": key, "kind": "fake", "version": "0.0.1"},
    }


def write_final_reports(root: Path, *, verdict: str = "transformer_better") -> Path:
    """Write a minimal but complete valid ``reports/final`` tree."""
    directory = root / "final"
    _write_json(
        directory / "metrics.json",
        {
            "schema_version": 1,
            "split": "test",
            "n_examples": 10,
            "models": {
                "baseline": _metric_block("baseline", 0.84, 0.85),
                "lora": _metric_block("lora", 0.86, 0.87),
            },
        },
    )
    _write_json(
        directory / "comparison.json",
        {
            "schema_version": 1,
            "split": "test",
            "n_examples": 10,
            "verdict": verdict,
            "delta": {
                "accuracy": {"estimate": 0.02, "lower": 0.005, "upper": 0.035},
                "macro_f1": {"estimate": 0.02, "lower": 0.005, "upper": 0.035},
            },
            "mcnemar": {"b": 6, "c": 0, "p": 0.03125},
            "verdict_sentences": {
                "en": "The LoRA transformer wins on this synthetic fixture.",
                "es": "El transformer con LoRA gana en este fixture sintético.",
            },
        },
    )
    _write_json(
        directory / "calibration_test.json",
        {
            "schema_version": 1,
            "split": "test",
            "n_examples": 10,
            "models": {
                key: {
                    "temperature": 1.5,
                    "review_threshold": 0.5,
                    "before": {"ece": 0.08, "nll": 0.4, "brier": 0.2},
                    "after": {"ece": 0.04, "nll": 0.35, "brier": 0.18},
                    "coverage_at_threshold": 0.8,
                    "selective_accuracy": 0.96,
                }
                for key in ("baseline", "lora")
            },
        },
    )
    _write_json(
        directory / "robustness.json",
        {
            "schema_version": 1,
            "split": "test",
            "n_examples": 10,
            "models": {
                key: {
                    "perturbations": [
                        {
                            "name": "uppercase",
                            "n_affected": 4,
                            "accuracy_before": 0.9,
                            "accuracy_after": 0.8,
                            "delta_accuracy": -0.1,
                            "delta_ci_lower": -0.2,
                            "delta_ci_upper": -0.0,
                            "mean_confidence_before": 0.8,
                            "mean_confidence_after": 0.7,
                        },
                        {
                            "name": "regionalize:ar",
                            "n_affected": 60,
                            "accuracy_before": 0.85,
                            "accuracy_after": 0.6,
                            "delta_accuracy": -0.25,
                            "delta_ci_lower": -0.35,
                            "delta_ci_upper": -0.15,
                            "mean_confidence_before": 0.8,
                            "mean_confidence_after": 0.6,
                        },
                    ]
                }
                for key in ("baseline", "lora")
            },
        },
    )
    _write_json(
        directory / "slices.json",
        {
            "schema_version": 1,
            "split": "test",
            "n_examples": 10,
            "models": {
                key: {
                    "slices": [
                        {
                            "name": "scenario:alarm",
                            "n": 10,
                            "accuracy": 0.9,
                            "accuracy_ci_lower": 0.8,
                            "accuracy_ci_upper": 1.0,
                            "macro_f1": None,
                            "macro_f1_ci_lower": None,
                            "macro_f1_ci_upper": None,
                        }
                    ]
                }
                for key in ("baseline", "lora")
            },
        },
    )
    _write_json(
        directory / "errors.json",
        {
            "schema_version": 1,
            "split": "test",
            "n_examples": 10,
            "n_errors": 2,
            "errors": [
                {
                    "example_id": str(index),
                    "text": f"texto {index}",
                    "true_intent": "alarm_set",
                    "baseline": {
                        "predicted_intent": "play_music",
                        "confidence": 0.55,
                        "correct": False,
                    },
                    "lora": {
                        "predicted_intent": "alarm_set",
                        "confidence": 0.8,
                        "correct": True,
                    },
                }
                for index in (1, 2)
            ],
        },
    )
    _write_csv(
        directory / "per_class.csv",
        ("model", "intent", "precision", "recall", "f1", "support"),
        [["lora", "alarm_set", 0.9, 0.8, 0.85, 5]],
    )
    _write_csv(
        directory / "confusion_matrix.csv",
        ("model", "true_intent", "predicted_intent", "count"),
        [["lora", "alarm_set", "alarm_set", 5]],
    )
    return directory


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_all_reports(root: Path) -> None:
    """Write the final, data-audit and baseline report families."""
    write_final_reports(root)
    _write_json(
        root / "data_audit.json",
        {
            "schema_version": 1,
            "tarball_sha256": "a" * 64,
            "dev_test_overlap_texts": 45,
            "dev_rows_in_train": 123,
            "test_rows_in_train": 152,
            "train_duplicate_groups": 264,
            "train_after": 10898,
            "processed_counts": {"train": 10898, "dev": 2033, "test": 2974},
            "drops": {"EXACT_DUPLICATE": 217},
        },
    )
    _write_json(
        root / "baseline" / "search.json",
        {
            "schema_version": 1,
            "configs": [
                {
                    "dev_accuracy": 0.85,
                    "dev_macro_f1": 0.84,
                    "n_features": 100,
                    "params": {},
                }
            ],
            "selection": {"winner": {}},
        },
    )
    _write_json(
        root / "baseline" / "dev_metrics.json",
        {
            "schema_version": 1,
            "split": "dev",
            "accuracy": 0.85,
            "macro_f1": 0.84,
            "weighted_f1": 0.85,
            "top3_accuracy": 0.94,
            "scenario_accuracy": 0.9,
        },
    )


@pytest.fixture
def final_reports(tmp_path: Path):
    """Return a writer callable for a valid final-report tree."""
    return lambda **kwargs: write_final_reports(tmp_path, **kwargs)


def write_experiment_reports(root: Path) -> None:
    """Write a minimal valid ``reports/experiments`` tree."""
    _write_json(
        root / "environment.json",
        {
            "schema_version": 1,
            "git_sha": "a" * 40,
            "uv_lock_sha256": "b" * 64,
            "cuda_available": True,
            "device_name": "NVIDIA GeForce RTX 4060",
            "driver_version": "620.70.06",
            "library_versions": {
                "torch": "2.15.0",
                "transformers": "4.57.6",
                "peft": "0.22.0",
                "optuna": "5.1.0",
                "mlflow": "3.18.0",
            },
            "stages_seconds": {"screening": 1.0},
        },
    )
    _write_json(
        root / "screening.json",
        {
            "schema_version": 1,
            "runs": [
                {
                    "backbone": "BSC-LT/MrBERT-es",
                    "seed": 13,
                    "dev_macro_f1": 0.84,
                    "dev_accuracy": 0.85,
                    "trainable_params": 100,
                    "epoch_seconds": 20.0,
                    "peak_vram_mib": 1500.0,
                    "cpu_latency_ms_p50": 15.5,
                }
            ],
        },
    )
    _write_json(
        root / "selection.json",
        {
            "schema_version": 1,
            "selected": "BSC-LT/MrBERT-es",
            "rule_branch": "mrbert_ahead",
            "per_backbone_means": {"BSC-LT/MrBERT-es": 0.84},
        },
    )
    _write_json(
        root / "full_ft_reference.json",
        {
            "schema_version": 1,
            "runs": [{"lr": 3e-5, "dev_macro_f1": 0.85}],
        },
    )
    _write_json(
        root / "search_trials.json",
        {
            "schema_version": 1,
            "trials": [{"number": 0, "state": "COMPLETE", "value": 0.83, "params": {"r": 16}}],
            "best": {"value": 0.83, "params": {"r": 16}},
        },
    )
    _write_json(
        root / "final_seeds.json",
        {
            "schema_version": 1,
            "seeds": [{"seed": 13, "dev_macro_f1": 0.84}],
            "selected_seed": 13,
            "dev_macro_f1_mean": 0.84,
            "dev_macro_f1_std": 0.0,
        },
    )
