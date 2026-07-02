"""Central configuration. Values can be overridden via environment variables.

All models are Google models served on Vertex AI in your own GCP project.
Authentication is Application Default Credentials (ADC): locally via
`gcloud auth application-default login`, on Cloud Run via the runtime
service account. No API keys are used anywhere.

A single fixed seed governs all stochastic steps (splitting, sampling) so the
evaluation entrypoint is reproducible from a clean clone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env once at import time (no-op if the file is absent).
load_dotenv(override=False)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    # --- GCP / Vertex AI ---
    # Project and location every Vertex call runs against. GOOGLE_CLOUD_PROJECT
    # is also set automatically on Cloud Run.
    project: str = field(default_factory=lambda: _env("GOOGLE_CLOUD_PROJECT", ""))
    location: str = field(
        default_factory=lambda: _env("GOOGLE_CLOUD_LOCATION", "us-central1")
    )
    # GCS bucket (name only, no gs://) used to stage supervised-tuning JSONL
    # files. Only needed when launching a fine-tuning job.
    gcs_bucket: str = field(default_factory=lambda: _env("TRUTHFULNESS_GCS_BUCKET", ""))

    # --- Models ---
    # Zero-shot and explainer default to the same base model that we fine-tune
    # (gemini-2.5-flash) so the zero-shot vs fine-tuned comparison isolates the
    # effect of tuning rather than a base-model difference.
    zero_shot_model: str = field(
        default_factory=lambda: _env("TRUTHFULNESS_ZERO_SHOT_MODEL", "gemini-2.5-flash")
    )
    explainer_model: str = field(
        default_factory=lambda: _env("TRUTHFULNESS_EXPLAINER_MODEL", "gemini-2.5-flash")
    )
    # Base model for Vertex AI supervised fine-tuning. As of July 2026 the
    # tunable Gemini models are gemini-2.5-flash, gemini-2.5-flash-lite and
    # gemini-2.5-pro (Gemini 3.x tuning is not yet available).
    fine_tune_base_model: str = field(
        default_factory=lambda: _env("TRUTHFULNESS_FT_BASE_MODEL", "gemini-2.5-flash")
    )
    # Resolved tuned-model endpoint (full resource name, e.g.
    # projects/P/locations/L/endpoints/123). Empty until fine_tune() completes,
    # or pin one here / via env to reuse an existing tuned model.
    fine_tuned_model: str = field(
        default_factory=lambda: _env("TRUTHFULNESS_FINE_TUNED_MODEL", "")
    )
    # Tuning epochs; 0 lets Vertex choose based on dataset size.
    ft_epochs: int = field(default_factory=lambda: _env_int("TRUTHFULNESS_FT_EPOCHS", 0))
    # Optional LoRA adapter size (1|2|4|8|16); 0 = provider default.
    ft_adapter_size: int = field(
        default_factory=lambda: _env_int("TRUTHFULNESS_FT_ADAPTER_SIZE", 0)
    )

    # Gemini 2.5 "thinking" budget for classification calls. 0 disables
    # thinking (cheapest, deterministic single-token answers). Only applied to
    # models that accept it (flash / flash-lite); -1 leaves the model default.
    thinking_budget: int = field(
        default_factory=lambda: _env_int("TRUTHFULNESS_THINKING_BUDGET", 0)
    )

    # --- Inference ---
    max_concurrency: int = field(
        default_factory=lambda: _env_int("TRUTHFULNESS_MAX_CONCURRENCY", 8)
    )
    request_timeout: int = field(
        default_factory=lambda: _env_int("TRUTHFULNESS_REQUEST_TIMEOUT", 60)
    )
    max_retries: int = field(default_factory=lambda: _env_int("TRUTHFULNESS_MAX_RETRIES", 5))

    # --- Reproducibility ---
    seed: int = field(default_factory=lambda: _env_int("TRUTHFULNESS_SEED", 123))

    # --- Splits (fractions of the full, de-duplicated dataset) ---
    test_size: float = 0.15
    val_size: float = 0.15  # carved out of the non-test portion for fine-tuning

    # --- Caching ---
    cache_dir: Path = field(
        default_factory=lambda: Path(_env("TRUTHFULNESS_CACHE_DIR", ".cache"))
    )
    use_cache: bool = field(
        default_factory=lambda: _env("TRUTHFULNESS_USE_CACHE", "1") == "1"
    )

    def require_project(self) -> str:
        if not self.project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set. Copy .env.example to .env and set "
                "your GCP project id, or export GOOGLE_CLOUD_PROJECT in your shell. "
                "Authenticate with `gcloud auth application-default login`."
            )
        return self.project

    def require_bucket(self) -> str:
        if not self.gcs_bucket:
            raise RuntimeError(
                "TRUTHFULNESS_GCS_BUCKET is not set. Fine-tuning stages its JSONL "
                "training data in a GCS bucket in your project; create one and set "
                "the bucket name (without gs://)."
            )
        return self.gcs_bucket


_CONFIG: Config | None = None


def get_config() -> Config:
    """Return a process-wide singleton config."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG
