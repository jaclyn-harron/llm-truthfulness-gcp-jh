"""Component 2: fine-tuned LLM predictor (Vertex AI supervised tuning of Gemini).

`fine_tune()` performs data preparation + a stratified train/val split, stages
the JSONL datasets in a GCS bucket, and launches a Vertex AI supervised
fine-tuning job on a Gemini base model (default: gemini-2.5-flash — the same
model the zero-shot predictor uses, so the comparison isolates the effect of
tuning). `predict()` is a drop-in replacement for the zero-shot predictor,
backed by the tuned model's endpoint.

The model is tuned to emit a single word (``true``/``false``). Vertex returns
token logprobs, so the reported confidence is a calibrated probability from
softmaxing the true/false token logprobs (with a plain-text fallback when
logprobs are unavailable).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from .base import PredictionOutput, attach_metrics, coerce_points
from .config import Config, get_config
from .data import points_from_frame, to_binary
from .llm import chat_json, client, map_concurrent, new_cache
from .prompts import ft_messages, ft_training_record


@dataclass
class FineTuneResult:
    job_name: str
    model: str | None  # tuned-model endpoint resource name once succeeded
    status: str
    base_model: str
    n_train: int
    n_val: int


_ADAPTER_SIZES = {
    1: "ADAPTER_SIZE_ONE",
    2: "ADAPTER_SIZE_TWO",
    4: "ADAPTER_SIZE_FOUR",
    8: "ADAPTER_SIZE_EIGHT",
    16: "ADAPTER_SIZE_SIXTEEN",
    32: "ADAPTER_SIZE_THIRTY_TWO",
}


def _adapter_size(n: int) -> str | None:
    """Map an integer adapter size onto the Vertex enum (0 = provider default)."""
    if not n:
        return None
    if n not in _ADAPTER_SIZES:
        raise ValueError(f"Unsupported adapter size {n}; choose one of {sorted(_ADAPTER_SIZES)}")
    return _ADAPTER_SIZES[n]


def _target_token(label_binary: bool) -> str:
    return "true" if label_binary else "false"


def _write_jsonl(df: pd.DataFrame, path: Path) -> int:
    """Write Vertex supervised-tuning JSONL (systemInstruction + contents)."""
    points = points_from_frame(df)
    targets = df["label_binary"].tolist()
    with path.open("w") as fh:
        for pt, y in zip(points, targets):
            rec = ft_training_record(pt, _target_token(bool(y)))
            fh.write(json.dumps(rec) + "\n")
    return len(points)


def _upload_to_gcs(local_path: Path, bucket_name: str, blob_name: str, project: str) -> str:
    from google.cloud import storage

    bucket = storage.Client(project=project).bucket(bucket_name)
    bucket.blob(blob_name).upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{blob_name}"


class FineTunedPredictor:
    """Vertex AI supervised-tuned Gemini classifier."""

    def __init__(self, config: Config | None = None, model: str | None = None):
        self.config = config or get_config()
        self.model = model or self.config.fine_tuned_model or None

    # ----------------------------- training ----------------------------------
    def fine_tune(
        self,
        training_dataset: pd.DataFrame,
        *,
        wait: bool = True,
        poll_seconds: int = 60,
        artifacts_dir: str | Path = "artifacts",
    ) -> FineTuneResult:
        """Prepare data, split, stage in GCS, and launch a Vertex tuning job.

        `training_dataset` is a DataFrame in data.csv format (must include
        `label`). The held-out test set is formed elsewhere and never passed in.
        """
        cfg = self.config
        cfg.require_project()
        bucket = cfg.require_bucket()

        df = training_dataset.copy()
        if "label_binary" not in df.columns:
            df["label"] = df["label"].astype(str).str.strip().str.lower()
            df["label_binary"] = to_binary(df["label"])

        train_df, val_df = train_test_split(
            df, test_size=cfg.val_size, random_state=cfg.seed, stratify=df["label_binary"]
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        out_dir = Path(artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        train_path = out_dir / "ft_train.jsonl"
        val_path = out_dir / "ft_val.jsonl"
        n_train = _write_jsonl(train_df, train_path)
        n_val = _write_jsonl(val_df, val_path)

        stamp = time.strftime("%Y%m%d-%H%M%S")
        train_uri = _upload_to_gcs(
            train_path, bucket, f"truthfulness/tuning/{stamp}/ft_train.jsonl", cfg.project
        )
        val_uri = _upload_to_gcs(
            val_path, bucket, f"truthfulness/tuning/{stamp}/ft_val.jsonl", cfg.project
        )

        from google.genai import types as genai_types

        tune_cfg = genai_types.CreateTuningJobConfig(
            tuned_model_display_name=f"truthfulness-{stamp}",
            validation_dataset=genai_types.TuningValidationDataset(gcs_uri=val_uri),
            epoch_count=cfg.ft_epochs or None,
            adapter_size=_adapter_size(cfg.ft_adapter_size),
        )
        job = client().tunings.tune(
            base_model=cfg.fine_tune_base_model,
            training_dataset=genai_types.TuningDataset(gcs_uri=train_uri),
            config=tune_cfg,
        )

        result = FineTuneResult(
            job_name=job.name,
            model=None,
            status=str(job.state),
            base_model=cfg.fine_tune_base_model,
            n_train=n_train,
            n_val=n_val,
        )
        (out_dir / "fine_tune_job.json").write_text(json.dumps(result.__dict__, indent=2))
        if not wait:
            return result

        result.model = self._poll(job.name, poll_seconds=poll_seconds)
        result.status = "succeeded"
        self.model = result.model
        (out_dir / "fine_tune_job.json").write_text(json.dumps(result.__dict__, indent=2))
        return result

    def _poll(self, job_name: str, *, poll_seconds: int) -> str:
        """Poll the tuning job until it ends; return the tuned-model endpoint."""
        while True:
            job = client().tunings.get(name=job_name)
            state = str(job.state or "")
            if "SUCCEEDED" in state:
                endpoint = getattr(job.tuned_model, "endpoint", None)
                if not endpoint:
                    raise RuntimeError(
                        f"Tuning job {job_name} succeeded but exposes no endpoint"
                    )
                return endpoint
            if any(bad in state for bad in ("FAILED", "CANCELLED", "EXPIRED")):
                raise RuntimeError(
                    f"Vertex tuning job {job_name} ended: {state} "
                    f"({getattr(job, 'error', None)})"
                )
            time.sleep(poll_seconds)

    # ----------------------------- inference ---------------------------------
    def predict(self, points, labels: Sequence | None = None) -> PredictionOutput:
        if not self.model:
            raise RuntimeError(
                "No fine-tuned model set. Run fine_tune() first, or pin "
                "TRUTHFULNESS_FINE_TUNED_MODEL to the tuned-model endpoint "
                "(projects/…/locations/…/endpoints/…)."
            )
        pts = coerce_points(points)
        cache = new_cache()

        def one(i: int) -> dict:
            res = chat_json(
                model=self.model,
                messages=ft_messages(pts[i]),
                temperature=0.0,
                max_tokens=5,
                logprobs=True,
                top_logprobs=5,
                cache=cache,
            )
            return _verdict_from_logprobs(res)

        parsed = map_concurrent(
            one, len(pts), max_workers=self.config.max_concurrency, desc="fine-tuned"
        )
        predictions = [p["verdict"] for p in parsed]
        confidence = [p["confidence"] for p in parsed]
        score_false = [p["p_false"] for p in parsed]
        out = PredictionOutput(
            predictions=predictions, confidence=confidence,
            score_false=score_false, raw=parsed,
        )
        return attach_metrics(out, labels)


def _verdict_from_logprobs(res: dict) -> dict:
    """Turn first-token output into a verdict + probability.

    With logprobs we softmax the `true`/`false` token logprobs for a calibrated
    probability. Without them we fall back to the emitted text with a
    deterministic confidence.
    """
    content = (res.get("content") or "").strip().lower()
    tops = res.get("top_logprobs") or []
    lp_true = lp_false = None
    for item in tops:
        tok = item["token"].strip().lower()
        if tok == "true" and lp_true is None:
            lp_true = item["logprob"]
        elif tok == "false" and lp_false is None:
            lp_false = item["logprob"]

    if lp_true is not None or lp_false is not None:
        a = math.exp(lp_true) if lp_true is not None else 1e-9
        b = math.exp(lp_false) if lp_false is not None else 1e-9
        p_true = a / (a + b)
    else:
        # No logprobs: scan the short completion for the answer word.
        # Check "false" first since "true" never appears inside "false".
        if "false" in content:
            p_true = 0.0
        elif "true" in content:
            p_true = 1.0
        else:
            p_true = 0.0  # malformed completion -> default to the majority-risk class

    verdict = p_true >= 0.5
    p_false = 1.0 - p_true
    confidence = p_true if verdict else p_false
    return {"verdict": bool(verdict), "confidence": float(confidence), "p_false": float(p_false)}
