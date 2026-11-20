"""Pydantic v2 request and response schemas for the public API (S14)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

__all__ = [
    "LabelsOut",
    "ModelBlockOut",
    "ModelInfoOut",
    "PredictRequest",
    "PredictResponse",
    "PredictionOut",
    "ProblemOut",
    "ScoredLabelOut",
]

_API_MAX_BATCH = 64
_API_MAX_CHARS = 512


class PredictRequest(BaseModel):
    """A bounded batch of utterances to classify."""

    model_config = ConfigDict(extra="forbid")

    texts: list[Annotated[str, StringConstraints(min_length=1, max_length=_API_MAX_CHARS)]] = Field(
        min_length=1, max_length=_API_MAX_BATCH
    )
    top_k: int = Field(default=3, ge=1, le=10)
    model: Literal["default", "baseline"] = "default"

    @field_validator("texts")
    @classmethod
    def _reject_blank_texts(cls, texts: list[str]) -> list[str]:
        cleaned: list[str] = []
        for text in texts:
            candidate = text.strip()
            if not candidate:
                raise ValueError("texts must not be blank")
            if len(candidate) > _API_MAX_CHARS:
                raise ValueError(f"each text must be at most {_API_MAX_CHARS} characters")
            cleaned.append(candidate)
        return cleaned


class ScoredLabelOut(BaseModel):
    """One candidate intent with its calibrated probability."""

    intent: str
    scenario: str
    probability: float


class PredictionOut(BaseModel):
    """The top prediction for one input text."""

    index: int
    intent: str
    scenario: str
    confidence: float
    needs_review: bool
    top_k: list[ScoredLabelOut]


class ModelBlockOut(BaseModel):
    """Cheap model identity embedded in prediction responses."""

    id: str
    kind: str
    version: str
    quantization: str | None
    temperature: float
    review_threshold: float


class PredictResponse(BaseModel):
    """The documented ``POST /v1/predict`` response."""

    schema_version: int = 1
    model: ModelBlockOut
    predictions: list[PredictionOut]
    latency_ms: float


class ModelInfoOut(BaseModel):
    """``GET /v1/model`` payload, read from the manifest and calibration."""

    id: str
    version: str
    kind: str
    quantization: str | None
    parameters: int | None
    license: str
    n_intents: int
    n_scenarios: int
    temperature: float
    review_threshold: float
    dev_ece_before: float | None
    dev_ece_after: float | None
    baseline_available: bool


class LabelsOut(BaseModel):
    """``GET /v1/labels`` payload."""

    intents: list[str]
    scenarios: list[str]
    n_intents: int
    n_scenarios: int


class ProblemOut(BaseModel):
    """RFC 7807 problem document returned by every error path."""

    type: str
    title: str
    status: int
    detail: str
