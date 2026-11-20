"""The ONNX classifier is CPU-only, order-preserving and torch-free."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stc.adapters.inference.onnx_classifier import OnnxClassifier, load_onnx_classifier
from stc.domain.errors import IntegrityError, InvalidInputError

REPO_ROOT = Path(__file__).resolve().parents[3]
TINY_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"


@pytest.fixture(scope="module")
def classifier() -> OnnxClassifier:
    return OnnxClassifier.from_bundle(TINY_BUNDLE)


def test_shapes_and_label_space(classifier: OnnxClassifier) -> None:
    scores = classifier.predict_scores(["enciende la luz", "ponme música de Juanes"])
    assert scores.shape == (2, classifier.labels.n_intents)
    assert scores.dtype == np.float32
    assert classifier.labels.n_intents == 60
    assert classifier.info.kind == "onnx-seqcls"


def test_batch_and_single_scores_are_identical(classifier: OnnxClassifier) -> None:
    texts = ["despiértame a las seis", "¿cómo está el clima en Quito?"]
    batched = classifier.predict_scores(texts)
    singles = np.vstack([classifier.predict_scores([text]) for text in texts])
    assert np.max(np.abs(batched - singles)) <= 1e-6


def test_short_and_long_texts_keep_input_order(classifier: OnnxClassifier) -> None:
    texts = ["luz", "recuérdame llamar a mi mamá mañana a las ocho en punto", "pizza"]
    batched = classifier.predict_scores(texts)
    for index in (0, 2):
        assert np.max(np.abs(batched[index] - classifier.predict_scores([texts[index]])[0])) <= 1e-6


def test_empty_input_is_rejected(classifier: OnnxClassifier) -> None:
    with pytest.raises(InvalidInputError):
        classifier.predict_scores([])


def test_alternative_model_file_loads(classifier: OnnxClassifier) -> None:
    variant = load_onnx_classifier(TINY_BUNDLE, classifier.model_path, threads=1)
    assert isinstance(variant, OnnxClassifier)
    assert variant.predict_scores(["hola"]).shape == (1, 60)


def test_threads_must_be_positive() -> None:
    with pytest.raises(InvalidInputError):
        OnnxClassifier.from_bundle(TINY_BUNDLE, threads=0)


def test_tampered_bundle_is_refused(tmp_path: Path) -> None:
    import shutil

    target = tmp_path / "bundle"
    shutil.copytree(TINY_BUNDLE, target)
    data = bytearray((target / "model.onnx").read_bytes())
    data[0] ^= 0xFF
    (target / "model.onnx").write_bytes(bytes(data))
    with pytest.raises(IntegrityError):
        OnnxClassifier.from_bundle(target)


def test_importing_the_inference_path_does_not_import_torch() -> None:
    code = (
        "import sys, stc.adapters.inference.onnx_classifier; "
        "assert 'torch' not in sys.modules, 'torch leaked into the runtime path'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
