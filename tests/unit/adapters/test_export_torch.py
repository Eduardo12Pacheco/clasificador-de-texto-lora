"""Torch-marked export tests: LoRA merge, ONNX parity and the D12 variants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from typer.testing import CliRunner

import stc.adapters.training.trainer as trainer_module
from stc.adapters.cli.app import app
from stc.adapters.export.onnx_export import OnnxExporter
from stc.adapters.export.quantization import QUANT_VARIANTS, build_variant
from stc.adapters.inference.bundle_io import verify_bundle
from stc.adapters.inference.onnx_classifier import OnnxClassifier
from stc.adapters.reporting.json_writer import JsonReportWriter
from stc.adapters.text.hf_tokenizer import HfTokenizer
from stc.application.use_cases.export_model import ExportModel
from stc.domain.splits import Split

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tiny-run")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(trainer_module, "device_for_training", lambda: torch.device("cpu"))
    try:
        result = CliRunner().invoke(
            app,
            [
                "train",
                "lora",
                "--config",
                "configs/smoke.toml",
                "--dataset",
                "fixture",
                "--out",
                str(out),
            ],
        )
    finally:
        patcher.undo()
    assert result.exit_code == 0, result.output
    return out / "seed-13"


def _dev_examples(n: int = 24):
    from stc.adapters.data.fixture_source import FixtureRepository

    return list(FixtureRepository().load(Split.DEV))[:n]


def test_export_merges_and_matches_torch(trained_run: Path, tmp_path: Path) -> None:
    exporter = OnnxExporter(base_model_loader=_tiny_loader)
    bundle = exporter.export(trained_run, tmp_path / "bundle")
    manifest = verify_bundle(bundle)
    assert manifest["kind"] == "onnx-seqcls"
    assert (bundle / "labels.json").is_file()
    texts = [example.text for example in _dev_examples(5)]
    torch_scores = exporter.score_logits(trained_run, texts)
    onnx_scores = OnnxClassifier.from_bundle(bundle).predict_scores(texts)
    assert torch_scores.shape == onnx_scores.shape == (5, 60)
    assert float(np.max(np.abs(torch_scores - onnx_scores))) <= 1e-4


def test_batch_and_single_inference_match(trained_run: Path, tmp_path: Path) -> None:
    exporter = OnnxExporter(base_model_loader=_tiny_loader)
    bundle = exporter.export(trained_run, tmp_path / "bundle")
    classifier = OnnxClassifier.from_bundle(bundle)
    texts = [example.text for example in _dev_examples(5)]
    batched = classifier.predict_scores(texts)
    singles = np.vstack([classifier.predict_scores([text]) for text in texts])
    assert float(np.max(np.abs(batched - singles))) <= 1e-4


def test_all_quantisation_variants_build(trained_run: Path, tmp_path: Path) -> None:
    exporter = OnnxExporter(base_model_loader=_tiny_loader)
    bundle = exporter.export(trained_run, tmp_path / "bundle")
    tokenizer = HfTokenizer.from_file(bundle / "tokenizer.json")
    texts = [example.text for example in _dev_examples(8)]
    for variant in QUANT_VARIANTS:
        target = build_variant(
            bundle / "model.onnx",
            variant,
            texts,
            tokenizer,
            tmp_path / f"{variant}.onnx",
        )
        assert target.is_file()
        assert target.stat().st_size > 0


def test_export_model_end_to_end(trained_run: Path, tmp_path: Path) -> None:
    dev = _dev_examples(24)
    outcome = ExportModel(
        OnnxExporter(base_model_loader=_tiny_loader),
        _Loader(),
        JsonReportWriter(),
    ).run(
        trained_run,
        tmp_path / "bundle",
        dev=dev,
        quantize="all",
        reports_dir=tmp_path / "reports",
        calibration_texts=[example.text for example in dev],
    )
    assert outcome.parity["passed"] is True
    assert outcome.shipped_format in {"fp32", *QUANT_VARIANTS}
    assert (tmp_path / "reports" / "export_parity.json").is_file()
    assert (tmp_path / "reports" / "quantization_dev.json").is_file()
    assert verify_bundle(outcome.bundle_dir)["kind"] == "onnx-seqcls"


class _Loader:
    def load(self, bundle_dir: Path, model_path: Path | None = None, *, threads=None):
        from stc.adapters.inference.onnx_classifier import load_onnx_classifier

        return load_onnx_classifier(bundle_dir, model_path, threads=threads)


def _tiny_loader(backbone: str, n_labels: int, tokenizer: object) -> object:
    from stc.adapters.training.model_factory import build_base_model

    return build_base_model(backbone, n_labels, tokenizer=tokenizer)


def test_export_cli_writes_parity_report(trained_run: Path, tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    result = CliRunner().invoke(
        app,
        [
            "export",
            "--run",
            str(trained_run),
            "--out",
            str(tmp_path / "bundle"),
            "--quantize",
            "none",
            "--dataset",
            "fixture",
            "--reports",
            str(reports),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "passed=True" in result.output
    payload = __import__("json").loads(
        (reports / "export_parity.json").read_text(encoding="utf-8")
    )
    assert payload["passed"] is True
    assert payload["agreement"] == 1.0
