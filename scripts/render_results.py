#!/usr/bin/env python
"""Build ``reports/summary.json`` from the final reports and render results.

The summary is the single source for the README results block, the standalone
results table and the model card, so a number can never be hand-edited in more
than one place.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

START_MARKER = "<!-- RESULTS:START -->"
END_MARKER = "<!-- RESULTS:END -->"
MODEL_LABELS = {"baseline": "TF-IDF baseline", "lora": "LoRA transformer"}
_INPUTS = ("metrics.json", "comparison.json", "calibration_test.json", "robustness.json")


def _load(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected an object")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name}: expected a number")
    return float(value)


def build_summary(final_dir: Path) -> dict[str, object]:
    """Read the final reports and aggregate the summary the docs consume."""
    metrics = _load(final_dir / "metrics.json")
    comparison = _load(final_dir / "comparison.json")
    calibration = _load(final_dir / "calibration_test.json")
    robustness = _load(final_dir / "robustness.json")

    model_metrics = _mapping(metrics.get("models"), "metrics.models")
    model_calibration = _mapping(calibration.get("models"), "calibration.models")
    model_robustness = _mapping(robustness.get("models"), "robustness.models")
    summary_models: dict[str, object] = {}
    for key in ("baseline", "lora"):
        block = _mapping(model_metrics.get(key), f"metrics.models.{key}")
        ci = _mapping(block.get("ci"), f"metrics.models.{key}.ci")
        accuracy_ci = _mapping(ci.get("accuracy"), f"metrics.models.{key}.ci.accuracy")
        macro_ci = _mapping(ci.get("macro_f1"), f"metrics.models.{key}.ci.macro_f1")
        cal = _mapping(model_calibration.get(key), f"calibration.models.{key}")
        before = _mapping(cal.get("before"), f"calibration.models.{key}.before")
        after = cal.get("after")
        after_mapping = _mapping(after, f"calibration.models.{key}.after") if after else {}
        perturbation_block = _mapping(model_robustness.get(key), f"robustness.models.{key}")
        perturbations = {}
        for row in perturbation_block.get("perturbations", []):
            row_mapping = _mapping(row, f"robustness.models.{key}.perturbation")
            perturbations[str(row_mapping["name"])] = {
                "accuracy_after": _number(row_mapping.get("accuracy_after"), "accuracy_after"),
                "accuracy_before": _number(row_mapping.get("accuracy_before"), "accuracy_before"),
                "delta_accuracy": _number(row_mapping.get("delta_accuracy"), "delta_accuracy"),
                "n_affected": int(_number(row_mapping.get("n_affected"), "n_affected")),
            }
        summary_models[key] = {
            "test": {
                "accuracy": _number(block.get("accuracy"), f"{key}.accuracy"),
                "accuracy_lower": _number(accuracy_ci.get("lower"), f"{key}.accuracy_lower"),
                "accuracy_upper": _number(accuracy_ci.get("upper"), f"{key}.accuracy_upper"),
                "macro_f1": _number(block.get("macro_f1"), f"{key}.macro_f1"),
                "macro_f1_lower": _number(macro_ci.get("lower"), f"{key}.macro_f1_lower"),
                "macro_f1_upper": _number(macro_ci.get("upper"), f"{key}.macro_f1_upper"),
                "scenario_accuracy": _number(block.get("scenario_accuracy"), f"{key}.scenario"),
                "top3_accuracy": _number(block.get("top3_accuracy"), f"{key}.top3"),
                "weighted_f1": _number(block.get("weighted_f1"), f"{key}.weighted_f1"),
            },
            "calibration": {
                "brier_after": (
                    _number(after_mapping.get("brier"), f"{key}.brier_after")
                    if after_mapping
                    else None
                ),
                "brier_before": _number(before.get("brier"), f"{key}.brier_before"),
                "coverage_at_threshold": _number(
                    cal.get("coverage_at_threshold"), f"{key}.coverage"
                ),
                "ece_after": (
                    _number(after_mapping.get("ece"), f"{key}.ece_after") if after_mapping else None
                ),
                "ece_before": _number(before.get("ece"), f"{key}.ece_before"),
                "nll_after": (
                    _number(after_mapping.get("nll"), f"{key}.nll_after") if after_mapping else None
                ),
                "nll_before": _number(before.get("nll"), f"{key}.nll_before"),
                "review_threshold": _number(cal.get("review_threshold"), f"{key}.tau"),
                "selective_accuracy": _number(
                    cal.get("selective_accuracy"), f"{key}.selective_accuracy"
                ),
                "temperature": _number(cal.get("temperature"), f"{key}.temperature"),
            },
            "robustness": perturbations,
        }

    comparison_block = _mapping(comparison, "comparison")
    delta = _mapping(comparison_block.get("delta"), "comparison.delta")
    accuracy_delta = _mapping(delta.get("accuracy"), "comparison.delta.accuracy")
    macro_delta = _mapping(delta.get("macro_f1"), "comparison.delta.macro_f1")
    mcnemar = _mapping(comparison_block.get("mcnemar"), "comparison.mcnemar")
    sentences = _mapping(comparison_block.get("verdict_sentences"), "verdict_sentences")
    return {
        "schema_version": 1,
        "models": summary_models,
        "comparison": {
            "verdict": str(comparison_block.get("verdict")),
            "verdict_sentence_en": str(sentences.get("en")),
            "verdict_sentence_es": str(sentences.get("es")),
            "delta_accuracy": _number(accuracy_delta.get("estimate"), "delta.accuracy"),
            "delta_accuracy_lower": _number(accuracy_delta.get("lower"), "delta.acc.lower"),
            "delta_accuracy_upper": _number(accuracy_delta.get("upper"), "delta.acc.upper"),
            "delta_macro_f1": _number(macro_delta.get("estimate"), "delta.macro_f1"),
            "delta_macro_f1_lower": _number(macro_delta.get("lower"), "delta.macro.lower"),
            "delta_macro_f1_upper": _number(macro_delta.get("upper"), "delta.macro.upper"),
            "mcnemar_b": int(_number(mcnemar.get("b"), "mcnemar.b")),
            "mcnemar_c": int(_number(mcnemar.get("c"), "mcnemar.c")),
            "mcnemar_p": _number(mcnemar.get("p"), "mcnemar.p"),
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _ci(model: Mapping[str, object], key: str) -> str:
    test = _mapping(model.get("test"), "model.test")
    lower_key = f"{key}_lower"
    upper_key = f"{key}_upper"
    estimate = _fmt(_number(test.get(key), key))
    lower = _fmt(_number(test.get(lower_key), lower_key))
    upper = _fmt(_number(test.get(upper_key), upper_key))
    return f"{estimate} [{lower}, {upper}]"


def render_table(summary: Mapping[str, object]) -> str:
    """Render the Markdown results table (same rows in README and docs)."""
    models = _mapping(summary.get("models"), "summary.models")
    lines = [
        "| Model | Test accuracy (95 % CI) | Test macro-F1 (95 % CI) | ECE before → after |",
        "|---|---|---|---|",
    ]
    for key in ("baseline", "lora"):
        model = _mapping(models.get(key), f"models.{key}")
        calibration = _mapping(model.get("calibration"), f"models.{key}.calibration")
        ece_before = _number(calibration.get("ece_before"), f"{key}.ece_before")
        ece_after = calibration.get("ece_after")
        ece_text = _fmt(ece_before)
        if ece_after is not None:
            ece_text += f" → {_fmt(_number(ece_after, f'{key}.ece_after'))}"
        lines.append(
            f"| {MODEL_LABELS[key]} | {_ci(model, 'accuracy')} | "
            f"{_ci(model, 'macro_f1')} | {ece_text} |"
        )
    return "\n".join(lines)


def render_block(summary: Mapping[str, object], language: str) -> str:
    """Render the table plus the generated verdict sentence for one language."""
    comparison = _mapping(summary.get("comparison"), "summary.comparison")
    sentence_key = "verdict_sentence_es" if language == "es" else "verdict_sentence_en"
    sentence = comparison.get(sentence_key)
    if not isinstance(sentence, str) or not sentence:
        raise ValueError(f"summary.comparison.{sentence_key} is empty")
    heading = "Resultados" if language == "es" else "Results"
    return f"### {heading}\n\n{render_table(summary)}\n\n{sentence}"


def _write_summary(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _is_outdated(summary_path: Path, final_dir: Path) -> bool:
    if not summary_path.is_file():
        return True
    summary_mtime = summary_path.stat().st_mtime_ns
    return any(
        (final_dir / name).is_file() and (final_dir / name).stat().st_mtime_ns > summary_mtime
        for name in _INPUTS
    )


def _replace_block(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError(f"{path}: expected exactly one RESULTS marker pair")
    before, remainder = text.split(START_MARKER, maxsplit=1)
    _, after = remainder.split(END_MARKER, maxsplit=1)
    updated = before + START_MARKER + "\n\n" + block.rstrip("\n") + "\n\n" + END_MARKER + after
    path.write_text(updated, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Render the summary, table and optional README blocks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("reports/summary.json"))
    parser.add_argument("--table", type=Path, default=Path("reports/results_table.md"))
    parser.add_argument("--final-dir", type=Path, default=None)
    parser.add_argument("--readme", type=Path, default=None)
    parser.add_argument("--readme-es", type=Path, default=None)
    args = parser.parse_args(argv)
    final_dir = args.final_dir if args.final_dir is not None else args.summary.parent / "final"
    try:
        if _is_outdated(args.summary, final_dir):
            summary = build_summary(final_dir)
            _write_summary(args.summary, summary)
        else:
            summary = _load(args.summary)
        table = render_table(summary)
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text(table + "\n", encoding="utf-8", newline="\n")
        if args.readme is not None:
            _replace_block(args.readme, render_block(summary, "en"))
        if args.readme_es is not None:
            _replace_block(args.readme_es, render_block(summary, "es"))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.summary} and {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
