"""Offline tests (no API calls): data framing, splits, metrics, parsing,
tuning-data format, and logprob-to-verdict conversion.

Run with:  PYTHONPATH=src pytest -q
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from truthfulness.base import parse_verdict_json
from truthfulness.data import BINARY_MAP, make_splits, to_binary
from truthfulness.fine_tuned import _adapter_size, _verdict_from_logprobs
from truthfulness.metrics import compute_metrics
from truthfulness.prompts import ft_training_record, render_point


def test_binary_map_is_balanced_split_of_ordinal():
    truthy = [k for k, v in BINARY_MAP.items() if v]
    assert truthy == ["true", "mostly-true", "half-true"]
    assert sum(BINARY_MAP.values()) == 3 and len(BINARY_MAP) == 6


def test_to_binary_accepts_strings_and_bools():
    assert to_binary(["true", "false", "extremely-false", True, False]) == [
        True, False, False, True, False,
    ]
    with pytest.raises(ValueError):
        to_binary(["not-a-label"])


def _toy_df(n=200):
    labels = (["true", "mostly-true", "half-true", "barely-true", "false", "extremely-false"]
              * (n // 6 + 1))[:n]
    return pd.DataFrame({
        "statement": [f"s{i}" for i in range(n)],
        "subjects": ["taxes"] * n,
        "speaker_name": ["x"] * n,
        "speaker_job": [None] * n,
        "speaker_state": [None] * n,
        "speaker_affiliation": ["democrat"] * n,
        "statement_context": ["a speech"] * n,
        "label": labels,
        "label_binary": to_binary(labels),
    })


def test_splits_are_disjoint_and_seeded():
    df = _toy_df()
    a = make_splits(df, seed=42, test_size=0.15, val_size=0.15)
    b = make_splits(df, seed=42, test_size=0.15, val_size=0.15)
    # Deterministic given the seed.
    assert list(a.test.statement) == list(b.test.statement)
    # No statement leaks across splits.
    s = {k: set(getattr(a, k).statement) for k in ["train", "val", "test"]}
    assert not (s["train"] & s["test"])
    assert not (s["train"] & s["val"])
    assert not (s["val"] & s["test"])
    assert len(a.train) + len(a.val) + len(a.test) == len(df)


def test_metrics_basic():
    m = compute_metrics(["true", "false", "true", "false"], [True, False, False, False])
    assert m["n"] == 4
    assert m["accuracy"] == pytest.approx(0.75)
    assert "macro_f1" in m and "f1_false" in m


def test_metrics_roc_and_brier_present_with_scores():
    m = compute_metrics(["true", "false"], [True, False], [0.1, 0.9])
    assert "roc_auc" in m and "brier" in m
    assert m["brier"] == pytest.approx(0.01, abs=1e-9)


def test_parse_verdict_json_variants():
    assert parse_verdict_json('{"verdict":"true","confidence":0.8}')["verdict"] is True
    assert parse_verdict_json('noise {"verdict":"false","confidence":0.9} tail')["verdict"] is False
    # Non-JSON fallback.
    assert parse_verdict_json("the statement is FALSE")["verdict"] is False
    # Confidence clamped to [0,1].
    assert 0.0 <= parse_verdict_json('{"verdict":"true","confidence":5}')["confidence"] <= 1.0


def test_render_point_omits_missing_and_expands_subjects():
    txt = render_point({"statement": "S", "subjects": "taxes$jobs", "speaker_job": None})
    assert "Statement: S" in txt
    assert "taxes, jobs" in txt
    assert "Speaker job" not in txt  # None field omitted


def test_ft_training_record_matches_vertex_tuning_schema():
    rec = ft_training_record({"statement": "S", "subjects": "taxes"}, "true")
    # Round-trips as a single JSONL line.
    rec = json.loads(json.dumps(rec))
    assert rec["systemInstruction"]["parts"][0]["text"]
    assert [c["role"] for c in rec["contents"]] == ["user", "model"]
    assert rec["contents"][0]["parts"][0]["text"].startswith("Statement: S")
    assert rec["contents"][1]["parts"][0]["text"] == "true"


def test_verdict_from_logprobs_softmax_and_fallback():
    import math

    # Calibrated path: softmax over true/false token logprobs.
    res = {"content": "true", "top_logprobs": [
        {"token": "true", "logprob": math.log(0.9)},
        {"token": "false", "logprob": math.log(0.1)},
    ]}
    out = _verdict_from_logprobs(res)
    assert out["verdict"] is True
    assert out["confidence"] == pytest.approx(0.9, abs=1e-6)
    assert out["p_false"] == pytest.approx(0.1, abs=1e-6)

    # Fallback path: no logprobs -> deterministic from text.
    assert _verdict_from_logprobs({"content": "false"})["verdict"] is False
    assert _verdict_from_logprobs({"content": "true"})["confidence"] == 1.0
    # Malformed completion defaults to the majority-risk class (False).
    assert _verdict_from_logprobs({"content": "?"})["verdict"] is False


def test_adapter_size_mapping():
    assert _adapter_size(0) is None
    assert _adapter_size(4) == "ADAPTER_SIZE_FOUR"
    with pytest.raises(ValueError):
        _adapter_size(3)
