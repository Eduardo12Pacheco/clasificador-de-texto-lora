"""ONNX Runtime int8 quantisation variants of the D12 matrix.

Every heavy import (``onnx``, ``onnxruntime.quantization``) happens inside the
functions so the base environment only needs ``onnxruntime`` to import this
module.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from stc.adapters.text.hf_tokenizer import HfTokenizer
from stc.domain.errors import InvalidInputError

__all__ = ["MLP_ONLY_EXCLUSIONS", "QUANT_VARIANTS", "build_variant", "excluded_nodes"]

QUANT_VARIANTS = (
    "int8_dynamic",
    "int8_dynamic_per_channel",
    "int8_dynamic_mlp_only",
    "int8_static_qdq",
)

MLP_ONLY_EXCLUSIONS = ("/attn/", "/head/", "/classifier/")
_STATIC_CALIBRATION_SAMPLES = 512


def excluded_nodes(model_path: Path, patterns: Sequence[str] = MLP_ONLY_EXCLUSIONS) -> list[str]:
    """Return graph node names that match any exclusion pattern."""
    import onnx

    model = onnx.load(str(model_path))
    return [
        node.name
        for node in model.graph.node
        if node.name and any(pattern in node.name for pattern in patterns)
    ]


class _CalibrationReader:
    """Feed fixed-width calibration batches to ``quantize_static``.

    ONNX Runtime's percentile calibrator stacks every batch it sees, so ragged
    dynamic shapes are not accepted; the reader pads each sample to the
    tokenizer's maximum length instead.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: HfTokenizer,
        *,
        batch_size: int = 8,
    ) -> None:
        import numpy as np

        self._texts = [str(text) for text in texts[:_STATIC_CALIBRATION_SAMPLES]]
        self._tokenizer = tokenizer
        self._batch_size = max(1, batch_size)
        self._width = tokenizer.max_length
        self._pad_id = tokenizer.pad_id
        self._np = np

    def get_next(self) -> dict[str, object] | None:
        if not self._texts:
            return None
        batch = self._texts[: self._batch_size]
        del self._texts[: self._batch_size]
        encoded = self._tokenizer.encode_batch(batch)
        width = self._width
        input_ids = self._np.full((len(batch), width), self._pad_id, dtype=self._np.int64)
        attention_mask = self._np.zeros((len(batch), width), dtype=self._np.int64)
        for row, length in enumerate(encoded.attention_mask.sum(axis=1)):
            take = int(min(length, width))
            input_ids[row, :take] = encoded.input_ids[row, :take]
            attention_mask[row, :take] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_variant(
    fp32_path: Path,
    variant: str,
    calibration_texts: Sequence[str],
    tokenizer: HfTokenizer,
    out_path: Path,
) -> Path:
    """Build one quantised ONNX file and return its path."""
    if variant not in QUANT_VARIANTS:
        raise InvalidInputError(
            f"unknown quantisation variant {variant!r}; expected {', '.join(QUANT_VARIANTS)}"
        )
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    source = Path(fp32_path)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if variant == "int8_static_qdq":
        # D12 selected percentile calibration, but per-channel percentile
        # histograms for a 150 M-parameter ModernBERT encoder peak at ~19 GiB
        # RSS (measured on T12) and the shared 31 GiB host OOM-kills the
        # process. MinMax calibration keeps the same QDQ graph, the same
        # per-channel setting and the same gate; only the calibration rule of
        # this non-shipping candidate changes, and ADR-0005 records it.
        quantize_static(
            str(source),
            str(target),
            _CalibrationReader(calibration_texts, tokenizer),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
        return target
    per_channel = variant != "int8_dynamic"
    nodes_to_exclude: list[str] = []
    if variant == "int8_dynamic_mlp_only":
        nodes_to_exclude = excluded_nodes(source)
    quantize_dynamic(
        str(source),
        str(target),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        nodes_to_exclude=nodes_to_exclude,
    )
    return target
