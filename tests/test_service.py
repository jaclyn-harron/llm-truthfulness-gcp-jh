"""Offline tests for the Part 2 service (no network): reconciliation logic,
payload schemas, and the A2A JSON extraction helper."""

from __future__ import annotations

from truthfulness_service.agents import _reconcile
from truthfulness_service.payloads import PredictResponse, VerifyResponse, VerifyResult


def test_reconcile_agreement():
    # When predictors agree, return that verdict with the higher confidence.
    assert _reconcile("fine_tuned", True, True, 0.6, 0.9) == (True, 0.9)
    assert _reconcile("confidence", False, False, 0.7, 0.5) == (False, 0.7)


def test_reconcile_disagreement_strategies():
    # Defer to fine-tuned (default).
    assert _reconcile("fine_tuned", True, False, 0.99, 0.51) == (False, 0.51)
    # Defer to zero-shot.
    assert _reconcile("zero_shot", True, False, 0.6, 0.9) == (True, 0.6)
    # Higher confidence wins.
    assert _reconcile("confidence", True, False, 0.9, 0.6) == (True, 0.9)
    assert _reconcile("confidence", True, False, 0.6, 0.9) == (False, 0.9)


def test_payload_roundtrip():
    pr = PredictResponse(predictions=[True], confidence=[0.8], score_false=[0.2])
    assert pr.model_dump()["predictions"] == [True]
    vr = VerifyResponse(
        results=[VerifyResult(prediction=True, agreement={"zero_shot": True, "fine_tuned": True},
                              confidence=0.8, explanation="x")],
        metrics={"n": 1},
    )
    d = vr.model_dump()
    assert d["results"][0]["agreement"]["fine_tuned"] is True
    assert d["metrics"]["n"] == 1


def test_extract_json_from_message():
    from types import SimpleNamespace as NS
    from truthfulness_service.a2a_base import _extract_json
    part = NS(root=NS(text='{"ok": 1}'))
    resp = NS(root=NS(result=NS(parts=[part], status=None)))
    assert _extract_json(resp) == {"ok": 1}
