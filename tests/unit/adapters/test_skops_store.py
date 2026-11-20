"""Skops persistence: allow-list, manifest hashes and prediction parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import skops.io as skops_io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from stc.adapters.models.skops_store import (
    SkopsBaselineStore,
    cast_linear_coefficients,
    is_allowed_type,
    load,
    save,
    sha256_file,
)
from stc.adapters.models.tfidf_linear import TfidfLinearClassifier, load_baseline
from stc.application.dto import ModelInfo
from stc.domain.errors import IntegrityError, UntrustedArtifactError
from stc.domain.labels import LabelSpace

pytestmark = pytest.mark.unit

INTENTS = ("alarm_cancel", "alarm_set", "play_music")


def make_classifier() -> TfidfLinearClassifier:
    pipeline = Pipeline(
        [
            ("features", TfidfVectorizer()),
            ("classifier", LinearSVC(C=0.5, random_state=13)),
        ]
    )
    texts = ["pon una alarma", "cancela la alarma", "pon música", "cancela música"]
    pipeline.fit(texts, ["alarm_set", "alarm_cancel", "play_music", "alarm_cancel"])
    return TfidfLinearClassifier(
        pipeline=pipeline,
        labels=LabelSpace.from_intents(INTENTS),
        info=ModelInfo(
            model_id="tfidf-linear-baseline",
            kind="tfidf-linear",
            version="0.1.0",
            quantization=None,
            parameters=12,
            license="MIT",
            n_intents=3,
            n_scenarios=3,
        ),
    )


def test_allow_list_accepts_sklearn_numpy_and_builtins() -> None:
    assert is_allowed_type("sklearn.pipeline.Pipeline")
    assert is_allowed_type("numpy.ndarray")
    assert is_allowed_type("scipy.sparse._csr.csr_matrix")
    assert is_allowed_type("builtins.list")
    assert not is_allowed_type("__main__.Evil")
    assert not is_allowed_type("os.system")
    assert not is_allowed_type("")


def test_round_trip_preserves_predictions(tmp_path: Path) -> None:
    classifier = make_classifier()
    store = SkopsBaselineStore()

    manifest = store.save(
        classifier,
        tmp_path,
        metadata={
            "winner": {"classifier": "linear_svc"},
            "dataset": {"dataset_id": "fixture"},
            "grid": {"quick": False},
            "n_configs": 42,
            "quick": False,
        },
    )

    loaded = TfidfLinearClassifier(
        pipeline=load(tmp_path / "baseline.skops"),
        labels=LabelSpace.from_intents(INTENTS),
        info=classifier.info,
    )
    texts = ["pon una alarma", "pon música", "cancela la alarma"]
    np.testing.assert_array_equal(classifier.predict_scores(texts), loaded.predict_scores(texts))
    assert manifest["kind"] == "tfidf-linear"
    assert manifest["winner"] == {"classifier": "linear_svc"}
    assert manifest["dataset"]["dataset_id"] == "fixture"
    assert manifest["n_configs"] == 42
    assert {entry["path"] for entry in manifest["files"]} == {
        "baseline.skops",
        "labels.json",
    }
    for entry in manifest["files"]:
        assert sha256_file(tmp_path / entry["path"]) == entry["sha256"]


def test_saved_layout_is_loadable_by_the_baseline_loader(tmp_path: Path) -> None:
    classifier = make_classifier()
    SkopsBaselineStore().save(classifier, tmp_path, metadata={"winner": {}})

    loaded = load_baseline(tmp_path)

    assert isinstance(loaded, TfidfLinearClassifier)
    assert loaded.labels.intents == INTENTS
    assert loaded.info.kind == "tfidf-linear"
    assert loaded.info.model_id == "tfidf-linear-baseline"


def test_tampered_skops_with_untrusted_type_is_rejected(tmp_path: Path) -> None:
    class Evil:
        pass

    path = tmp_path / "baseline.skops"
    skops_io.dump({"payload": Evil()}, path)

    with pytest.raises(UntrustedArtifactError):
        load(path)


def test_manifest_hash_mismatch_is_an_integrity_error(tmp_path: Path) -> None:
    store = SkopsBaselineStore()
    classifier = make_classifier()
    store.save(classifier, tmp_path, metadata={"winner": {}})
    original = (tmp_path / "baseline.skops").read_bytes()
    (tmp_path / "baseline.skops").write_bytes(original + b"tampered")

    with pytest.raises(IntegrityError):
        load_baseline(tmp_path)


def test_wrong_kind_in_manifest_is_rejected(tmp_path: Path) -> None:
    store = SkopsBaselineStore()
    store.save(make_classifier(), tmp_path, metadata={"winner": {}})
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["kind"] = "onnx-seqcls"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IntegrityError):
        load_baseline(tmp_path)


def test_missing_artifact_names_the_fix_command(tmp_path: Path) -> None:
    from stc.domain.errors import ArtifactNotFoundError

    with pytest.raises(ArtifactNotFoundError, match="stc train baseline"):
        load_baseline(tmp_path / "missing")


def test_save_and_load_helpers_round_trip(tmp_path: Path) -> None:
    pipeline = Pipeline(
        [("features", TfidfVectorizer()), ("classifier", LinearSVC(C=0.5, random_state=13))]
    )
    pipeline.fit(["uno", "dos", "tres"], ["alarm_set", "alarm_cancel", "play_music"])
    path = tmp_path / "pipeline.skops"

    save(pipeline, path)

    loaded = load(path)
    assert hasattr(loaded, "predict")


def test_cast_linear_coefficients_uses_float32() -> None:
    pipeline = Pipeline(
        [("features", TfidfVectorizer()), ("classifier", LinearSVC(C=0.5, random_state=13))]
    )
    pipeline.fit(["uno", "dos", "tres"], ["alarm_set", "alarm_cancel", "play_music"])
    classifier = pipeline.named_steps["classifier"]
    classifier.coef_ = np.ones((3, 5), dtype=np.float64)

    assert cast_linear_coefficients(pipeline) is True
    assert classifier.coef_.dtype == np.float32
    assert cast_linear_coefficients(object()) is False


def test_store_rejects_a_classifier_without_a_pipeline(tmp_path: Path) -> None:
    from stc.domain.errors import InvalidInputError

    store = SkopsBaselineStore()

    with pytest.raises(InvalidInputError):
        store.save(object(), tmp_path, metadata={})  # type: ignore[arg-type]
