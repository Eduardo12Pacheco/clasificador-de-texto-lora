"""HTTP routes of the public API (S14)."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from stc.adapters.api.errors import ProblemError
from stc.adapters.api.provider import ClassifierProvider, LoadedModel
from stc.adapters.api.schemas import (
    LabelsOut,
    ModelInfoOut,
    PredictRequest,
    PredictResponse,
)
from stc.application.use_cases.predict_texts import PredictTexts
from stc.domain.errors import ArtifactNotFoundError

__all__ = ["WEB_ROOT", "router"]

WEB_ROOT = Path(__file__).resolve().parents[1] / "web" / "static"

router = APIRouter()


def _provider(request: Request) -> ClassifierProvider:
    return cast(ClassifierProvider, request.app.state.provider)


def _default_model(request: Request) -> LoadedModel:
    return cast(LoadedModel, request.app.state.default_model)


def _baseline_model(request: Request) -> LoadedModel:
    if not bool(request.app.state.baseline_available):
        raise ProblemError(
            404,
            "baseline not available",
            "the installed bundle has no baseline; run: stc model pull",
        )
    cached = request.app.state.baseline_model
    if cached is None:
        try:
            cached = _provider(request).load_baseline()
        except ArtifactNotFoundError as error:
            raise ProblemError(404, "baseline not available", error.message) from error
        request.app.state.baseline_model = cached
    return cast(LoadedModel, cached)


def _optional_float(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


@router.get("/healthz")
def healthz(request: Request) -> dict[str, object]:
    """Liveness probe that also reports whether the model finished loading."""
    loaded = getattr(request.app.state, "default_model", None) is not None
    return {"status": "ok", "model_loaded": loaded}


@router.get("/v1/model", response_model=ModelInfoOut)
def model_info(request: Request) -> ModelInfoOut:
    """Describe the served artifact and its calibration."""
    loaded = _default_model(request)
    info = loaded.classifier.info
    calibration = loaded.calibration
    return ModelInfoOut(
        id=info.model_id,
        version=info.version,
        kind=info.kind,
        quantization=info.quantization,
        parameters=info.parameters,
        license=info.license,
        n_intents=info.n_intents,
        n_scenarios=info.n_scenarios,
        temperature=loaded.classifier.calibration.temperature,
        review_threshold=loaded.classifier.calibration.review_threshold,
        dev_ece_before=_optional_float(calibration, "ece_dev_before"),
        dev_ece_after=_optional_float(calibration, "ece_dev_after"),
        baseline_available=bool(request.app.state.baseline_available),
    )


@router.get("/v1/labels", response_model=LabelsOut)
def labels(request: Request) -> LabelsOut:
    """Expose the fixed label space the served model was trained on."""
    space = _default_model(request).classifier.labels
    return LabelsOut(
        intents=list(space.intents),
        scenarios=list(space.scenarios),
        n_intents=space.n_intents,
        n_scenarios=space.n_scenarios,
    )


@router.post("/v1/predict", response_model=PredictResponse)
def predict(request: Request, payload: PredictRequest) -> PredictResponse:
    """Classify a bounded batch and return calibrated top-k predictions."""
    request.state.n_texts = len(payload.texts)
    loaded = _baseline_model(request) if payload.model == "baseline" else _default_model(request)
    started = time.perf_counter()
    document = PredictTexts().run(loaded.classifier, payload.texts, top_k=payload.top_k)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return PredictResponse.model_validate({**document, "latency_ms": latency_ms})


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the bundled intent-classifier UI."""
    return FileResponse(WEB_ROOT / "index.html", media_type="text/html")
