"""Export a trained run, measure ONNX parity and apply the D12 gate (S11)."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stc.application.dto import ExportOutcome
from stc.application.ports import JsonSink, ModelExporter, OnnxModelLoader
from stc.domain.errors import InvalidInputError
from stc.domain.examples import Example
from stc.domain.metrics import accuracy, confusion_matrix, macro_f1, present_labels

__all__ = ["QUANT_VARIANTS", "ExportModel"]

# D12 candidate matrix; every variant is evaluated on dev and only the smallest
# one passing the pre-registered gate may ship.
QUANT_VARIANTS: tuple[str, ...] = (
    "int8_dynamic",
    "int8_dynamic_per_channel",
    "int8_dynamic_mlp_only",
    "int8_static_qdq",
)

PARITY_MAX_LOGIT_DIFF = 5e-3


def _metrics(scores: np.ndarray, y_true: np.ndarray, n_labels: int) -> dict[str, float]:
    predicted = np.argmax(scores, axis=1).astype(np.int64)
    matrix = confusion_matrix(y_true, predicted, n_labels)
    return {
        "accuracy": accuracy(y_true, predicted),
        "macro_f1": macro_f1(matrix, present_labels(y_true)),
    }


class ExportModel:
    """Orchestrate export, torch/ONNX parity and the quantisation gate."""

    def __init__(
        self,
        exporter: ModelExporter,
        loader: OnnxModelLoader,
        writer: JsonSink,
    ) -> None:
        self._exporter = exporter
        self._loader = loader
        self._writer = writer

    def run(
        self,
        run_dir: Path,
        out_dir: Path,
        *,
        dev: Sequence[Example],
        quantize: str = "none",
        reports_dir: Path | None = None,
        calibration_texts: Sequence[str] | None = None,
    ) -> ExportOutcome:
        """Export the run, evaluate parity and (optionally) the D12 matrix."""
        if quantize not in {"none", "all"}:
            raise InvalidInputError("quantize must be 'none' or 'all'")
        if not dev:
            raise InvalidInputError("export needs dev examples for the parity/Gate reports")

        bundle_dir = self._exporter.export(Path(run_dir), Path(out_dir))
        labels = self._loader.load(bundle_dir).labels
        texts = [example.text for example in dev]
        y_true = np.asarray(
            [labels.intent_index(example.intent) for example in dev], dtype=np.int64
        )

        torch_scores = np.asarray(self._exporter.score_logits(Path(run_dir), texts))
        onnx_classifier = self._loader.load(bundle_dir)
        fp32_scores = np.asarray(onnx_classifier.predict_scores(texts), dtype=np.float32)
        if torch_scores.shape != fp32_scores.shape:
            raise InvalidInputError(
                f"torch scores {torch_scores.shape} and ONNX scores {fp32_scores.shape} differ"
            )
        max_diff = float(np.max(np.abs(torch_scores - fp32_scores)))
        agreement = float(
            np.mean(np.argmax(torch_scores, axis=1) == np.argmax(fp32_scores, axis=1))
        )
        parity: dict[str, object] = {
            "schema_version": 1,
            "split": "dev",
            "n_examples": len(dev),
            "max_abs_logit_diff": max_diff,
            "agreement": agreement,
            "max_allowed_logit_diff": PARITY_MAX_LOGIT_DIFF,
            "passed": bool(max_diff <= PARITY_MAX_LOGIT_DIFF and agreement == 1.0),
        }
        fp32_metrics = _metrics(fp32_scores, y_true, labels.n_intents)

        variants: list[dict[str, object]] = []
        passing: list[tuple[int, str, Path]] = []
        shipped_format = "fp32"
        if quantize == "all":
            calibration = list(calibration_texts or texts)[:512]
            with tempfile.TemporaryDirectory() as scratch:
                for variant in QUANT_VARIANTS:
                    variant_path = self._exporter.build_variant(
                        bundle_dir,
                        variant,
                        calibration,
                        Path(scratch) / f"variant.{variant}.onnx",
                    )
                    variant_scores = np.asarray(
                        self._loader.load(bundle_dir, variant_path, threads=1).predict_scores(
                            texts
                        ),
                        dtype=np.float32,
                    )
                    gate = dict(self._exporter.gate(fp32_scores, variant_scores, y_true.tolist()))
                    row: dict[str, object] = {
                        "name": variant,
                        "size_bytes": variant_path.stat().st_size,
                        **_metrics(variant_scores, y_true, labels.n_intents),
                        **gate,
                    }
                    variants.append(row)
                    if row.get("gate_passed") or row.get("passed"):
                        passing.append((variant_path.stat().st_size, variant, variant_path))
                if passing:
                    _, best, best_path = min(passing)
                    self._exporter.ship_variant(bundle_dir, best, best_path)
                    shipped_format = best
        quantization_payload: dict[str, object] = {
            "schema_version": 1,
            "split": "dev",
            "n_examples": len(dev),
            "candidates": list(QUANT_VARIANTS),
            "fp32": {"name": "fp32", **fp32_metrics},
            "variants": variants,
            "shipped_format": shipped_format,
        }
        if reports_dir is not None:
            self._writer.write_json(Path(reports_dir) / "export_parity.json", parity)
            self._writer.write_json(
                Path(reports_dir) / "quantization_dev.json", quantization_payload
            )
        return ExportOutcome(
            bundle_dir=bundle_dir,
            parity=parity,
            quantization=quantization_payload,
            shipped_format=shipped_format,
        )
