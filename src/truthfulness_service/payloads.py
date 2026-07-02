"""Pydantic schemas for the JSON payloads carried over A2A messages and returned
by MCP tools. Agents exchange these as JSON text in A2A message parts."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Point(BaseModel):
    statement: str
    subjects: Optional[str] = None
    speaker_name: Optional[str] = None
    speaker_job: Optional[str] = None
    speaker_state: Optional[str] = None
    speaker_affiliation: Optional[str] = None
    statement_context: Optional[str] = None


class PredictRequest(BaseModel):
    points: list[Point]
    labels: Optional[list[str]] = None


class PredictResponse(BaseModel):
    predictions: list[bool]
    confidence: list[float]
    score_false: list[float]
    metrics: Optional[dict] = None


class ExplainRequest(BaseModel):
    points: list[Point]
    # If verdicts are supplied, explanations are conditioned on them (the
    # orchestrator passes the final consensus). Otherwise the named model runs.
    verdicts: Optional[list[bool]] = None
    confidences: Optional[list[float]] = None
    model: str = "fine_tuned"
    labels: Optional[list[str]] = None


class ExplanationItem(BaseModel):
    prediction: bool
    confidence: float
    key_factors: list[str] = Field(default_factory=list)
    rationale: str = ""
    explanation: str = ""
    consistent: bool = True


class ExplainResponse(BaseModel):
    items: list[ExplanationItem]
    metrics: Optional[dict] = None


class VerifyResult(BaseModel):
    prediction: bool
    agreement: dict  # {"zero_shot": bool, "fine_tuned": bool}
    confidence: float
    explanation: str


class VerifyResponse(BaseModel):
    results: list[VerifyResult]
    metrics: Optional[dict] = None
    trace_id: Optional[str] = None


def points_to_dicts(points: list[Point]) -> list[dict]:
    return [p.model_dump() for p in points]
