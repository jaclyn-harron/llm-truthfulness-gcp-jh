"""The agents. Each wraps a Part 1 capability and consumes MCP tools as a client.
The orchestrator delegates to the others over A2A and reconciles their outputs.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from a2a.types import AgentCard

from .a2a_base import JsonAgentExecutor, build_card, discover_and_call
from .config import ServiceConfig
from .mcp_client import call_mcp_tool
from .payloads import VerifyResponse, VerifyResult

logger = logging.getLogger("truthfulness.agents")
VERSION = "0.1.0"


# ------------------------------- predictor agents ----------------------------

def _predictor_handler(cfg: ServiceConfig, tool: str):
    async def handler(payload: dict) -> dict:
        res = await call_mcp_tool(
            cfg.mcp_url, tool,
            {"points": payload.get("points", []), "labels": payload.get("labels")},
        )
        return res
    return handler


def zero_shot_card(cfg: ServiceConfig) -> AgentCard:
    return build_card(
        name="Zero-shot Predictor Agent",
        description="Predicts statement truthfulness with a pure zero-shot LLM (Component 1).",
        url=cfg.resolved_public_url(), version=VERSION,
        skill_id="predict_zero_shot", skill_name="Zero-shot truthfulness prediction",
        skill_description="Given a batch of statements, return True/False verdicts (+metrics).",
        tags=["truthfulness", "prediction", "zero-shot"],
    )


def fine_tuned_card(cfg: ServiceConfig) -> AgentCard:
    return build_card(
        name="Fine-tuned Predictor Agent",
        description="Predicts truthfulness with the Vertex-tuned Gemini model (Component 2).",
        url=cfg.resolved_public_url(), version=VERSION,
        skill_id="predict_fine_tuned", skill_name="Fine-tuned truthfulness prediction",
        skill_description="Given a batch of statements, return True/False verdicts (+metrics).",
        tags=["truthfulness", "prediction", "fine-tuned"],
    )


# ------------------------------- explainer agent -----------------------------

def _explainer_handler(cfg: ServiceConfig):
    async def handler(payload: dict) -> dict:
        res = await call_mcp_tool(
            cfg.mcp_url, "explain",
            {
                "points": payload.get("points", []),
                "verdicts": payload.get("verdicts"),
                "confidences": payload.get("confidences"),
                "model": payload.get("model", "fine_tuned"),
                "labels": payload.get("labels"),
            },
        )
        return res
    return handler


def explainer_card(cfg: ServiceConfig) -> AgentCard:
    return build_card(
        name="Explainer Agent",
        description="Generates faithful, verdict-consistent explanations (Component 3).",
        url=cfg.resolved_public_url(), version=VERSION,
        skill_id="explain", skill_name="Explain truthfulness verdicts",
        skill_description="Explain the factors driving each verdict for a batch of statements.",
        tags=["truthfulness", "explanation"],
    )


# ------------------------------- orchestrator --------------------------------

def _reconcile(strategy: str, zs: bool, ft: bool, zs_c: float, ft_c: float) -> tuple[bool, float]:
    if zs == ft:
        return zs, max(zs_c, ft_c)
    if strategy == "zero_shot":
        return zs, zs_c
    if strategy == "confidence":
        return (zs, zs_c) if zs_c >= ft_c else (ft, ft_c)
    return ft, ft_c  # default: defer to the fine-tuned (trained) model


def _orchestrator_handler(cfg: ServiceConfig):
    async def handler(payload: dict) -> dict:
        trace_id = payload.get("trace_id") or uuid.uuid4().hex
        points = payload.get("points", [])
        labels = payload.get("labels")
        logger.info("[%s] orchestrator: fan-out to predictor agents (n=%s)", trace_id, len(points))

        # 1) Delegate prediction to both predictor agents over A2A, in parallel.
        sub = {"points": points, "labels": labels, "trace_id": trace_id}
        zs_res, ft_res = await asyncio.gather(
            discover_and_call(cfg.zero_shot_url, sub),
            discover_and_call(cfg.fine_tuned_url, sub),
        )
        zs_pred, zs_conf = zs_res["predictions"], zs_res["confidence"]
        ft_pred, ft_conf = ft_res["predictions"], ft_res["confidence"]
        ft_scores = ft_res.get("score_false", [1.0 - c if v else c for v, c in zip(ft_pred, ft_conf)])

        # 2) Reconcile disagreements into a final verdict per statement.
        finals, fin_conf, agreements = [], [], []
        for i in range(len(points)):
            v, c = _reconcile(cfg.reconcile, zs_pred[i], ft_pred[i], zs_conf[i], ft_conf[i])
            finals.append(v); fin_conf.append(c)
            agreements.append({"zero_shot": bool(zs_pred[i]), "fine_tuned": bool(ft_pred[i])})
        logger.info("[%s] orchestrator: reconciled (strategy=%s)", trace_id, cfg.reconcile)

        # 3) Ask the explainer agent (A2A) to explain the FINAL consensus verdicts.
        exp_res = await discover_and_call(cfg.explainer_url, {
            "points": points, "verdicts": finals, "confidences": fin_conf,
            "labels": labels, "trace_id": trace_id,
        })
        items = exp_res.get("items", [])

        results = []
        for i in range(len(points)):
            expl = items[i]["explanation"] if i < len(items) else ""
            results.append(VerifyResult(
                prediction=bool(finals[i]), agreement=agreements[i],
                confidence=float(fin_conf[i]), explanation=expl,
            ).model_dump())

        # 4) Aggregate metrics via the shared MCP metric tool (orchestrator is
        #    also an MCP client), when gold labels were supplied.
        metrics = None
        if labels:
            score_false = [(1.0 - c) if v else c for v, c in zip(finals, fin_conf)]
            metrics = await call_mcp_tool(cfg.mcp_url, "compute_metrics", {
                "y_true": labels, "y_pred": finals, "y_score_false": score_false,
            })
        logger.info("[%s] orchestrator: complete", trace_id)
        return VerifyResponse(results=results, metrics=metrics, trace_id=trace_id).model_dump()
    return handler


def orchestrator_card(cfg: ServiceConfig) -> AgentCard:
    return build_card(
        name="Truthfulness Orchestrator",
        description="Entry point: fans out to predictor agents over A2A, reconciles, "
                    "and returns a verdict + explanation (+ metrics) per statement.",
        url=cfg.resolved_public_url(), version=VERSION,
        skill_id="verify", skill_name="Verify statement truthfulness",
        skill_description="Verify a batch of statements end to end and return verdicts, "
                          "agreement, explanations and aggregate metrics.",
        tags=["truthfulness", "orchestration", "verification"],
    )


# ------------------------------- role registry -------------------------------

def build_role(cfg: ServiceConfig):
    """Return (card, executor) for the configured ROLE."""
    role = cfg.role
    if role == "zero_shot":
        return zero_shot_card(cfg), JsonAgentExecutor(role, _predictor_handler(cfg, "predict_zero_shot"))
    if role == "fine_tuned":
        return fine_tuned_card(cfg), JsonAgentExecutor(role, _predictor_handler(cfg, "predict_fine_tuned"))
    if role == "explainer":
        return explainer_card(cfg), JsonAgentExecutor(role, _explainer_handler(cfg))
    if role == "orchestrator":
        return orchestrator_card(cfg), JsonAgentExecutor(role, _orchestrator_handler(cfg))
    raise ValueError(f"Unknown A2A ROLE: {role}")
