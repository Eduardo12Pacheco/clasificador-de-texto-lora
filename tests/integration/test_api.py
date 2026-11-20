"""Integration tests for the FastAPI service on the committed tiny bundle."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from stc.adapters.api.app import create_app
from stc.adapters.api.provider import BootstrapClassifierProvider
from stc.config import Settings

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "tiny_bundle"
PROBLEM = "application/problem+json"


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    settings = Settings(home=tmp_path, model=str(TINY_BUNDLE))
    provider = BootstrapClassifierProvider(settings, default_selector=str(TINY_BUNDLE))
    return TestClient(create_app(settings, provider))


def test_healthz_reports_the_loaded_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_comes_from_the_manifest_and_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/v1/model")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "stc-tiny-test"
    assert payload["version"] == "0.0.1"
    assert payload["kind"] == "onnx-seqcls"
    assert payload["quantization"] == "fp32"
    assert payload["n_intents"] == 60
    assert payload["temperature"] == 1.5
    assert payload["review_threshold"] == 0.5
    assert payload["baseline_available"] is False


def test_labels_expose_intents_and_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/v1/labels")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["intents"]) == payload["n_intents"] == 60
    assert len(payload["scenarios"]) == payload["n_scenarios"]


def test_predict_preserves_the_input_order_and_top_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/predict",
            json={"texts": ["apaga la luz", "enciende la luz"], "top_k": 2},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["model"]["id"] == "stc-tiny-test"
    assert [item["index"] for item in payload["predictions"]] == [0, 1]
    assert all(len(item["top_k"]) == 2 for item in payload["predictions"])
    assert payload["latency_ms"] >= 0.0


@pytest.mark.parametrize(
    "body",
    [
        {"texts": [f"texto {index}" for index in range(65)]},
        {"texts": ["x" * 513]},
        {"texts": ["hola"], "top_k": 11},
        {"texts": ["   "]},
        {"texts": []},
    ],
)
def test_invalid_predict_bodies_return_a_problem_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
) -> None:
    with _client(tmp_path, monkeypatch) as client:
        response = client.post("/v1/predict", json=body)

    assert response.status_code == 422
    assert response.headers["content-type"].startswith(PROBLEM)
    payload = response.json()
    assert {"type", "title", "status", "detail"} <= set(payload)
    assert payload["status"] == 422


def test_baseline_request_without_a_baseline_returns_404_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/predict",
            json={"texts": ["hola"], "model": "baseline"},
        )

    assert response.status_code == 404
    assert response.headers["content-type"].startswith(PROBLEM)


def test_security_headers_are_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/healthz")

    assert response.headers["content-security-policy"] == "default-src 'self'"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_request_id_is_echoed_and_generated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _client(tmp_path, monkeypatch) as client:
        echoed = client.get("/healthz", headers={"X-Request-ID": "test-request-id"})
        generated = client.get("/healthz")

    assert echoed.headers["x-request-id"] == "test-request-id"
    assert generated.headers["x-request-id"]
    assert generated.headers["x-request-id"] != "test-request-id"


def test_access_logging_never_records_input_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    marker = "texto-secreto-que-no-debe-registrarse"
    caplog.set_level(logging.DEBUG)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post("/v1/predict", json={"texts": [marker]})

    assert response.status_code == 200
    assert marker not in caplog.text
