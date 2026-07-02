"""MCP tool server exposing the Part 1 capabilities as reusable tools.

Tools (callable by our agents or any generic MCP client, e.g. Claude/Cursor):
  - predict_zero_shot(points, labels?)   -> predictions + metrics
  - predict_fine_tuned(points, labels?)  -> predictions + metrics
  - explain(points, verdicts?, ...)      -> per-point explanations + metrics
  - compute_metrics(y_true, y_pred, ...) -> metric bundle   (shared support tool)
  - dataset_sample(n, seed?)             -> sample rows      (shared support tool)

The fine-tuned predictor is a Vertex AI supervised-tuned Gemini model. Its
endpoint must be provided via TRUTHFULNESS_FINE_TUNED_MODEL (run the tuning job
once with `python -m truthfulness.evaluate` and pin the resulting endpoint).
All Vertex access uses Application Default Credentials — on Cloud Run, the
runtime service account; no API keys.
"""

from __future__ import annotations

import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from truthfulness.config import get_config
from truthfulness.data import load_dataset, make_splits, points_from_frame, labels_from_frame
from truthfulness.explainer import Explainer
from truthfulness.fine_tuned import FineTunedPredictor
from truthfulness.metrics import compute_metrics as _compute_metrics
from truthfulness.zero_shot import ZeroShotPredictor

mcp = FastMCP(
    "truthfulness-tools",
    host=os.environ.get("HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", "8085")),
)

_cfg = get_config()
_zero_shot = ZeroShotPredictor(_cfg)
_explainer = Explainer(_cfg)
_fine_tuned = FineTunedPredictor(_cfg)

DATA_PATH = os.environ.get("DATA_PATH", "data.csv")


def _get_fine_tuned() -> FineTunedPredictor:
    if not _fine_tuned.model:
        raise RuntimeError(
            "TRUTHFULNESS_FINE_TUNED_MODEL is not set. Run the tuning job once "
            "(python -m truthfulness.evaluate --data data.csv) and set the env "
            "var to the tuned-model endpoint (projects/…/endpoints/…)."
        )
    return _fine_tuned


def _predict_payload(predictor, points: list[dict], labels: Optional[list[str]]) -> dict:
    out = predictor.predict(points, labels=labels)
    return {
        "predictions": out.predictions,
        "confidence": out.confidence,
        "score_false": out.score_false,
        "metrics": out.metrics,
    }


@mcp.tool()
def predict_zero_shot(points: list[dict], labels: Optional[list[str]] = None) -> dict:
    """Zero-shot LLM truthfulness prediction for a batch of statements.

    points: list of dicts with keys statement, subjects, speaker_name,
    speaker_job, speaker_state, speaker_affiliation, statement_context.
    labels: optional gold six-way labels; if given, metrics are returned."""
    return _predict_payload(_zero_shot, points, labels)


@mcp.tool()
def predict_fine_tuned(points: list[dict], labels: Optional[list[str]] = None) -> dict:
    """Fine-tuned (Vertex AI supervised-tuned Gemini) truthfulness prediction
    for a batch. Same I/O contract as predict_zero_shot."""
    return _predict_payload(_get_fine_tuned(), points, labels)


@mcp.tool()
def explain(
    points: list[dict],
    verdicts: Optional[list[bool]] = None,
    confidences: Optional[list[float]] = None,
    model: str = "fine_tuned",
    labels: Optional[list[str]] = None,
) -> dict:
    """Explain truthfulness verdicts for a batch.

    If `verdicts` are supplied, explanations are conditioned on them (use this to
    explain a final consensus). Otherwise the named `model` ("zero_shot" or
    "fine_tuned") is run and its predictions explained."""
    if verdicts is not None:
        out = _explainer.explain_verdicts(
            points, verdicts=verdicts, confidences=confidences, labels=labels
        )
    else:
        predictor = _get_fine_tuned() if model == "fine_tuned" else _zero_shot
        out = _explainer.explain(predictor, points, labels=labels)
    return {"items": [e for e in out.to_records()], "metrics": out.metrics}


@mcp.tool()
def compute_metrics(
    y_true: list[str], y_pred: list[bool], y_score_false: Optional[list[float]] = None
) -> dict:
    """Shared metric tool: accuracy, balanced accuracy, macro-F1, precision/recall/F1
    for the False (misinformation) class, ROC-AUC and Brier (when scores given)."""
    return _compute_metrics(y_true, y_pred, y_score_false)


@mcp.tool()
def dataset_sample(n: int = 5, seed: Optional[int] = None) -> dict:
    """Shared dataset tool: return a random sample of n statements (with gold
    labels) from the held-out test split — useful for demos and retrieval."""
    df = load_dataset(DATA_PATH)
    splits = make_splits(df, seed=_cfg.seed, test_size=_cfg.test_size, val_size=_cfg.val_size)
    sample = splits.test.sample(n=min(n, len(splits.test)), random_state=seed or _cfg.seed)
    return {
        "points": points_from_frame(sample),
        "labels": labels_from_frame(sample),
    }


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
