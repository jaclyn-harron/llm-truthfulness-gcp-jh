"""Shared result types and parsing helpers for predictors."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

from .data import points_from_frame
from .metrics import compute_metrics


@dataclass
class PredictionOutput:
    """Output of a predictor over a batch of points.

    Attributes
    ----------
    predictions : bool per point (True = truthful).
    confidence  : model's confidence in its own verdict, 0..1.
    score_false : probability the statement is False (positive class), used for
                  ROC-AUC / calibration. Equals (1 - confidence) when the
                  verdict is True, and `confidence` when the verdict is False.
    metrics     : populated only when gold labels are supplied.
    """

    predictions: list[bool]
    confidence: list[float]
    score_false: list[float]
    raw: list[dict] = field(default_factory=list)
    metrics: dict | None = None

    def to_records(self) -> list[dict]:
        return [
            {"prediction": p, "confidence": c, "score_false": s}
            for p, c, s in zip(self.predictions, self.confidence, self.score_false)
        ]


@runtime_checkable
class Predictor(Protocol):
    """Both ZeroShotPredictor and FineTunedPredictor satisfy this, so they are
    interchangeable wherever a predictor is expected (e.g. the Explainer)."""

    def predict(
        self, points, labels: Sequence | None = None
    ) -> PredictionOutput: ...


def coerce_points(points) -> list[dict]:
    """Accept a DataFrame or a list of dict-like points; return list[dict]."""
    if isinstance(points, pd.DataFrame):
        return points_from_frame(points)
    out: list[dict] = []
    for p in points:
        if isinstance(p, dict):
            out.append(p)
        elif hasattr(p, "_asdict"):
            out.append(dict(p._asdict()))
        else:
            raise TypeError(f"Unsupported point type: {type(p)}")
    return out


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_TRUE_RE = re.compile(r"\btrue\b", re.IGNORECASE)
_FALSE_RE = re.compile(r"\bfalse\b", re.IGNORECASE)


def parse_verdict_json(content: str) -> dict:
    """Robustly parse a {"verdict":..., "confidence":...} payload.

    Falls back to keyword scanning if the model returned non-JSON text.
    """
    content = (content or "").strip()
    data: dict = {}
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        m = _JSON_RE.search(content)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = {}

    verdict = data.get("verdict")
    if isinstance(verdict, str):
        is_true = verdict.strip().lower().startswith("t")
    elif isinstance(verdict, bool):
        is_true = verdict
    else:
        # Fallback: scan free text.
        t, f = bool(_TRUE_RE.search(content)), bool(_FALSE_RE.search(content))
        is_true = t and not f

    conf = data.get("confidence")
    try:
        conf = float(conf)
        conf = min(max(conf, 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.6  # neutral-ish default when the model omitted confidence

    out = {"verdict": bool(is_true), "confidence": conf}
    # Carry through explainer fields if present.
    for k in ("key_factors", "rationale", "verdict_agrees"):
        if k in data:
            out[k] = data[k]
    return out


def attach_metrics(out: PredictionOutput, labels: Sequence | None) -> PredictionOutput:
    if labels is not None:
        out.metrics = compute_metrics(labels, out.predictions, out.score_false)
    return out
