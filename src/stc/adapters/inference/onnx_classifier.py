"""CPU-only ONNX Runtime adapter for the ``Classifier`` port (S12).

This module must stay importable without ``torch`` or ``onnx``: it only needs
``onnxruntime`` and the Rust ``tokenizers`` adapter shared with training.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from stc.adapters.inference.bundle_io import (
    bundle_info,
    load_labels,
    model_file,
    read_manifest,
    verify_bundle,
)
from stc.adapters.text.hf_tokenizer import HfTokenizer
from stc.application.dto import ModelInfo
from stc.domain.errors import IntegrityError, InvalidInputError
from stc.domain.labels import LabelSpace

__all__ = ["OnnxClassifier", "load_onnx_classifier"]


class OnnxClassifier:
    """Score utterances with a bundle ONNX file through ONNX Runtime."""

    def __init__(
        self,
        model_path: Path,
        tokenizer: HfTokenizer,
        labels: LabelSpace,
        info: ModelInfo,
        *,
        threads: int | None = None,
    ) -> None:
        if threads is not None and threads < 1:
            raise InvalidInputError("threads must be a positive integer")
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        input_names = {item.name for item in self._session.get_inputs()}
        missing = {"input_ids", "attention_mask"} - input_names
        if missing:
            raise IntegrityError("ONNX model is missing input(s): " + ", ".join(sorted(missing)))
        self._tokenizer = tokenizer
        self._labels = labels
        self._info = info
        self._model_path = Path(model_path)

    @classmethod
    def from_files(
        cls,
        model_path: Path,
        tokenizer_path: Path,
        labels: LabelSpace,
        info: ModelInfo,
        *,
        threads: int | None = None,
    ) -> OnnxClassifier:
        """Build a classifier from explicit artifact paths (used by the exporter)."""
        tokenizer = HfTokenizer.from_file(tokenizer_path)
        return cls(model_path, tokenizer, labels, info, threads=threads)

    @classmethod
    def from_bundle(
        cls,
        bundle: Path,
        *,
        threads: int | None = None,
        verify: bool = True,
    ) -> OnnxClassifier:
        """Load ``bundle``, verifying every manifest hash by default."""
        root = Path(bundle)
        manifest = verify_bundle(root) if verify else read_manifest(root)
        if manifest["kind"] != "onnx-seqcls":
            raise InvalidInputError(f"bundle {root} is not an onnx-seqcls bundle")
        labels = load_labels(root)
        tokenizer = HfTokenizer.from_file(root / "tokenizer.json")
        return cls(
            model_file(root, manifest),
            tokenizer,
            labels,
            bundle_info(root, manifest, labels),
            threads=threads,
        )

    @property
    def info(self) -> ModelInfo:
        """Describe the loaded bundle."""
        return self._info

    @property
    def labels(self) -> LabelSpace:
        """The label space the logits are ordered by."""
        return self._labels

    @property
    def model_path(self) -> Path:
        """Absolute path of the ONNX file this classifier scored with."""
        return self._model_path

    def predict_scores(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return one row of fp32 logits per utterance, preserving input order."""
        items = list(texts)
        if not items:
            raise InvalidInputError("prediction needs at least one text")
        encoded = self._tokenizer.encode_batch(items)
        outputs = self._session.run(
            ["logits"],
            {
                "input_ids": encoded.input_ids,
                "attention_mask": encoded.attention_mask,
            },
        )
        scores = np.asarray(outputs[0], dtype=np.float32)
        expected = (len(items), self._labels.n_intents)
        if scores.shape != expected:
            raise IntegrityError(f"ONNX model returned shape {scores.shape}, expected {expected}")
        return scores


def load_onnx_classifier(
    bundle: Path,
    model_path: Path | None = None,
    *,
    threads: int | None = None,
) -> OnnxClassifier:
    """Load a bundle, optionally scoring with an alternative ONNX file.

    The alternative path is used by the quantisation gate to evaluate a variant
    without touching the shipped ``model.onnx``.
    """
    classifier = OnnxClassifier.from_bundle(bundle, threads=threads)
    if model_path is None or Path(model_path) == classifier.model_path:
        return classifier
    # Alternative model files (quantisation variants) are produced outside the
    # bundle, so only the shared metadata is reused here.
    from stc.adapters.inference.bundle_io import bundle_info, load_labels, read_manifest

    manifest = read_manifest(bundle)
    labels = load_labels(bundle)
    return OnnxClassifier.from_files(
        Path(model_path),
        Path(bundle) / "tokenizer.json",
        labels,
        bundle_info(Path(bundle), manifest, labels),
        threads=threads,
    )
