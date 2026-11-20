"""The committed OpenAPI snapshot must match the live app."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO_ROOT / "docs" / "openapi.json"
SCRIPT = REPO_ROOT / "scripts" / "export_openapi.py"


def test_committed_openapi_matches_the_generated_document(tmp_path: Path) -> None:
    generated = tmp_path / "openapi.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(generated)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert generated.read_bytes() == SNAPSHOT.read_bytes()


def test_snapshot_documents_every_public_route() -> None:
    document = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    assert document["info"]["version"] == "0.1.0"
    assert {"/healthz", "/v1/model", "/v1/labels", "/v1/predict"} <= set(document["paths"])
    request = document["components"]["schemas"]["PredictRequest"]
    assert request["properties"]["texts"]["maxItems"] == 64
    assert request["properties"]["top_k"]["maximum"] == 10
