"""Build ``reports/summary.json`` from the final reports and render results.

The summary is the single source for the README results block, the standalone
results table and the model card, so a number can never be hand-edited in more
than one place.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

START_MARKER = "<!-- RESULTS:START -->"
END_MARKER = "<!-- RESULTS:END -->"
MODEL_LABELS = {"baseline": "TF-IDF baseline", "lora": "LoRA transformer"}
_INPUTS = ("metrics.json", "comparison.json", "calibration_test.json", "robustness.json")

__all__ = [
    "END_MARKER",
    "START_MARKER",
    "build_summary",
    "bundle_block",
    "render_block",
    "render_results",
    "render_table",
    "runtime_block",
    "runtime_paragraph",
]


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


def _optional_mapping(path: Path) -> Mapping[str, object] | None:
    """Load an optional JSON object, returning ``None`` when the file is absent."""
    if not path.is_file():
        return None
    return _load(path)


def runtime_block(final_dir: Path) -> dict[str, object] | None:
    """Return the CPU latency summary from ``benchmark.json`` when present."""
    payload = _optional_mapping(final_dir / "benchmark.json")
    if payload is None:
        return None
    threads = payload.get("threads")
    if not isinstance(threads, list):
        raise ValueError("benchmark.threads: expected a list")
    rows = [row for row in threads if isinstance(row, dict)]
    if not rows:
        return None
    chosen = rows[-1]
    for row in rows:
        if row.get("threads") == 4:
            chosen = row
            break
    block: dict[str, object] = {
        "threads": int(_number(chosen.get("threads"), "benchmark.threads")),
        "p50_ms": _number(chosen.get("p50_ms"), "benchmark.p50_ms"),
        "p95_ms": _number(chosen.get("p95_ms"), "benchmark.p95_ms"),
    }
    cpu_model = payload.get("cpu_model")
    if isinstance(cpu_model, str) and cpu_model:
        block["cpu_model"] = cpu_model
    version = payload.get("onnxruntime_version")
    if isinstance(version, str) and version:
        block["onnxruntime_version"] = version
    throughput = chosen.get("batch32_throughput_per_s")
    if not isinstance(throughput, bool) and isinstance(throughput, int | float):
        block["batch32_throughput_per_s"] = float(throughput)
    return block


def bundle_block(registry_path: Path | None) -> dict[str, object] | None:
    """Return the published bundle identity and size from the model registry."""
    if registry_path is None:
        return None
    payload = _optional_mapping(registry_path)
    if payload is None:
        return None
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        return None
    entry = models[0]
    if not isinstance(entry, dict):
        raise ValueError(f"{registry_path}: models[0] must be an object")
    return {
        "name": str(entry.get("name")),
        "version": str(entry.get("version")),
        "size_bytes": int(_number(entry.get("size_bytes"), "registry.size_bytes")),
    }


def _attach_optional(
    summary: dict[str, object], final_dir: Path, registry_path: Path | None
) -> bool:
    """Attach runtime and bundle metadata, returning whether anything changed."""
    changed = False
    if "runtime" not in summary:
        runtime = runtime_block(final_dir)
        if runtime is not None:
            summary["runtime"] = runtime
            changed = True
    if "bundle" not in summary:
        bundle = bundle_block(registry_path)
        if bundle is not None:
            summary["bundle"] = bundle
            changed = True
    return changed


def build_summary(final_dir: Path, *, registry_path: Path | None = None) -> dict[str, object]:
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
        raw_perturbations = perturbation_block.get("perturbations", [])
        if not isinstance(raw_perturbations, list):
            raise ValueError("robustness perturbations: expected a list")
        for row in raw_perturbations:
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
    summary: dict[str, object] = {
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
    _attach_optional(summary, final_dir, registry_path)
    return summary


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
    environment = runtime_paragraph(summary, language)
    body = f"### {heading}\n\n{render_table(summary)}\n\n{sentence}"
    if environment:
        body += f"\n\n{environment}"
    return body


def runtime_paragraph(summary: Mapping[str, object], language: str) -> str:
    """Render the generated CPU-latency and bundle-size line, when reported."""
    parts: list[str] = []
    runtime = summary.get("runtime")
    if isinstance(runtime, dict):
        p50 = _number(runtime.get("p50_ms"), "runtime.p50_ms")
        p95 = _number(runtime.get("p95_ms"), "runtime.p95_ms")
        threads = int(_number(runtime.get("threads"), "runtime.threads"))
        throughput = runtime.get("batch32_throughput_per_s")
        cpu_model = runtime.get("cpu_model")
        if language == "es":
            text = (
                f"Inferencia en CPU: {p50:.1f} ms p50 / {p95:.1f} ms p95 por frase "
                f"con {threads} hilos"
            )
            if not isinstance(throughput, bool) and isinstance(throughput, int | float):
                text += f" (rendimiento batch-32 de {float(throughput):.1f} frases/s)"
            if isinstance(cpu_model, str) and cpu_model:
                text += f" en {cpu_model}"
        else:
            text = (
                f"CPU inference: {p50:.1f} ms p50 / {p95:.1f} ms p95 per utterance "
                f"at {threads} threads"
            )
            if not isinstance(throughput, bool) and isinstance(throughput, int | float):
                text += f" (batch-32 throughput {float(throughput):.1f} utterances/s)"
            if isinstance(cpu_model, str) and cpu_model:
                text += f" on {cpu_model}"
        parts.append(text + ".")
    bundle = summary.get("bundle")
    if isinstance(bundle, dict):
        megabytes = _number(bundle.get("size_bytes"), "bundle.size_bytes") / 1_000_000
        if language == "es":
            parts.append(f"Bundle ONNX fp32 de {megabytes:.1f} MB.")
        else:
            parts.append(f"fp32 ONNX bundle: {megabytes:.1f} MB.")
    return " ".join(parts)


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


def render_results(
    *,
    summary_path: Path = Path("reports/summary.json"),
    table_path: Path = Path("reports/results_table.md"),
    final_dir: Path | None = None,
    readme: Path | None = None,
    readme_es: Path | None = None,
    registry_path: Path | None = None,
) -> tuple[Path, Path]:
    """Rebuild the summary when stale and render the table and README blocks."""
    resolved_final = final_dir if final_dir is not None else summary_path.parent / "final"
    if _is_outdated(summary_path, resolved_final):
        summary = build_summary(resolved_final, registry_path=registry_path)
        _write_summary(summary_path, summary)
    else:
        summary = _load(summary_path)
        if _attach_optional(summary, resolved_final, registry_path):
            _write_summary(summary_path, summary)
    table = render_table(summary)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table + "\n", encoding="utf-8", newline="\n")
    if readme is not None:
        _replace_block(readme, render_block(summary, "en"))
    if readme_es is not None:
        _replace_block(readme_es, render_block(summary, "es"))
    return summary_path, table_path
