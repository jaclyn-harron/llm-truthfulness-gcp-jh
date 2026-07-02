"""Component 1: zero-shot, pure-LLM predictor (no training of any kind)."""

from __future__ import annotations

from typing import Sequence

from .base import PredictionOutput, attach_metrics, coerce_points, parse_verdict_json
from .config import Config, get_config
from .llm import chat_json, map_concurrent, new_cache
from .prompts import predict_messages


class ZeroShotPredictor:
    """Prompts an off-the-shelf Gemini model with a statement's attributes and
    asks for a truthful/not-truthful verdict plus a confidence. No fine-tuning,
    no training — it relies purely on the model's prior knowledge.
    """

    def __init__(self, config: Config | None = None, model: str | None = None):
        self.config = config or get_config()
        self.model = model or self.config.zero_shot_model

    def predict(self, points, labels: Sequence | None = None) -> PredictionOutput:
        """Predict True/False for a batch of points.

        Efficient over a set: requests are issued concurrently and cached.
        If `labels` is provided, performance metrics are attached.
        """
        pts = coerce_points(points)
        cache = new_cache()

        def one(i: int) -> dict:
            res = chat_json(
                model=self.model,
                messages=predict_messages(pts[i]),
                temperature=0.0,
                max_tokens=40,
                json_mode=True,
                cache=cache,
            )
            return parse_verdict_json(res["content"])

        parsed = map_concurrent(
            one, len(pts), max_workers=self.config.max_concurrency, desc="zero-shot"
        )

        predictions = [p["verdict"] for p in parsed]
        confidence = [p["confidence"] for p in parsed]
        # Probability of the positive class (False).
        score_false = [
            (1.0 - c) if v else c for v, c in zip(predictions, confidence)
        ]
        out = PredictionOutput(
            predictions=predictions,
            confidence=confidence,
            score_false=score_false,
            raw=parsed,
        )
        return attach_metrics(out, labels)
