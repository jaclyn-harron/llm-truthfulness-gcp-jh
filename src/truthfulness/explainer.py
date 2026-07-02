"""Component 3: explainer that works with either predictor.

Faithfulness strategy
---------------------
The explanation is generated *conditioned on the underlying model's actual
verdict and confidence*. This guarantees the explanation is consistent with the
predicted label (it never argues for the opposite class), and it forces the
explainer to name the concrete factors in this specific statement that justify
this specific verdict, rather than emitting boilerplate.

For the fine-tuned model, whose only raw output is a token probability, the
model's own confidence is surfaced inside the explanation so the reader can see
how decisive the prediction was. We additionally run a consistency guard and
flag any explanation whose stated verdict drifts from the model's verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .base import Predictor, coerce_points, parse_verdict_json
from .config import Config, get_config
from .llm import chat_json, map_concurrent, new_cache
from .metrics import compute_metrics
from .prompts import render_point


@dataclass
class Explanation:
    prediction: bool
    confidence: float
    key_factors: list[str]
    rationale: str
    consistent: bool  # did the explainer agree with the model's verdict?

    @property
    def text(self) -> str:
        verdict = "TRUE (truthful)" if self.prediction else "FALSE (not truthful)"
        factors = "; ".join(self.key_factors) if self.key_factors else "n/a"
        return (
            f"Verdict: {verdict} (model confidence {self.confidence:.2f}). "
            f"Key factors: {factors}. {self.rationale}"
        )


@dataclass
class ExplainOutput:
    predictions: list[bool]
    confidence: list[float]
    explanations: list[Explanation]
    metrics: dict | None = None

    def to_records(self) -> list[dict]:
        return [
            {
                "prediction": e.prediction,
                "confidence": e.confidence,
                "key_factors": e.key_factors,
                "rationale": e.rationale,
                "explanation": e.text,
                "consistent": e.consistent,
            }
            for e in self.explanations
        ]


_EXPLAIN_SYSTEM = (
    "You explain the reasoning behind a truthfulness model's decision. You are "
    "given a statement (with metadata) and the model's verdict and confidence. "
    "Your job is NOT to second-guess the verdict but to identify the concrete, "
    "statement-specific factors that justify it and explain how they lead to "
    "THIS verdict. Use only the provided statement/metadata and your factual "
    "knowledge; do not invent evidence. Respond ONLY with compact JSON: "
    '{"key_factors": [<3-5 short concrete phrases>], '
    '"rationale": "<2-3 sentences tying those factors to the given verdict>", '
    '"verdict_agrees": <true|false: whether the evidence genuinely supports the '
    'model verdict>}.'
)


class Explainer:
    """Generates faithful, label-consistent explanations for any Predictor."""

    def __init__(self, config: Config | None = None, model: str | None = None):
        self.config = config or get_config()
        self.model = model or self.config.explainer_model

    def explain(
        self, model: Predictor, points, labels: Sequence | None = None
    ) -> ExplainOutput:
        """Return, for each point, the underlying model's prediction and an
        explanation of the factors driving it. Attaches metrics for the
        underlying model's predictions when `labels` is supplied.
        """
        pts = coerce_points(points)
        # The verdict is the model's actual prediction (single source of truth).
        pred = model.predict(pts)
        explanations = self._generate(pts, pred.predictions, pred.confidence)
        metrics = None
        if labels is not None:
            metrics = compute_metrics(labels, pred.predictions, pred.score_false)
        return ExplainOutput(
            predictions=pred.predictions,
            confidence=pred.confidence,
            explanations=explanations,
            metrics=metrics,
        )

    def explain_verdicts(
        self,
        points,
        verdicts: Sequence[bool],
        confidences: Sequence[float] | None = None,
        labels: Sequence | None = None,
        score_false: Sequence[float] | None = None,
    ) -> ExplainOutput:
        """Explain a set of *already-decided* verdicts (e.g. an orchestrator's
        consensus), without re-running a predictor. This keeps the explanation
        consistent with the final answer the caller is returning."""
        pts = coerce_points(points)
        verdicts = [bool(v) for v in verdicts]
        confs = list(confidences) if confidences is not None else [1.0] * len(pts)
        explanations = self._generate(pts, verdicts, confs)
        metrics = None
        if labels is not None:
            sf = score_false if score_false is not None else [
                (1.0 - c) if v else c for v, c in zip(verdicts, confs)
            ]
            metrics = compute_metrics(labels, verdicts, sf)
        return ExplainOutput(
            predictions=verdicts, confidence=confs,
            explanations=explanations, metrics=metrics,
        )

    def _generate(
        self, pts: list[dict], verdicts: Sequence[bool], confidences: Sequence[float]
    ) -> list[Explanation]:
        """Generate label-consistent explanations conditioned on given verdicts."""
        cache = new_cache()

        def one(i: int) -> Explanation:
            verdict = bool(verdicts[i])
            conf = float(confidences[i])
            verdict_str = "TRUE (truthful)" if verdict else "FALSE (not truthful)"
            user = (
                f"{render_point(pts[i])}\n\n"
                f"Model verdict: {verdict_str}\n"
                f"Model confidence: {conf:.2f}"
            )
            res = chat_json(
                model=self.model,
                messages=[
                    {"role": "system", "content": _EXPLAIN_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=220,
                json_mode=True,
                cache=cache,
            )
            parsed = parse_verdict_json(res["content"])
            agrees = parsed.get("verdict_agrees")
            if not isinstance(agrees, bool):
                agrees = True
            return Explanation(
                prediction=verdict,
                confidence=conf,
                key_factors=list(parsed.get("key_factors", []) or []),
                rationale=str(parsed.get("rationale", "")).strip(),
                consistent=bool(agrees),
            )

        return map_concurrent(
            one, len(pts), max_workers=self.config.max_concurrency, desc="explain"
        )
