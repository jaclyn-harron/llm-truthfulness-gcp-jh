"""Truthfulness: LLM-based binary truthfulness classification on Vertex AI.

All models are Google models (Gemini) served from your own GCP project.

Public API:
    ZeroShotPredictor   - pure-LLM, no training (Component 1)
    FineTunedPredictor  - Vertex AI supervised-tuned Gemini (Component 2)
    Explainer           - faithful per-prediction explanations (Component 3)
    load_dataset, to_binary, make_splits - data utilities
    compute_metrics     - shared metric computation
"""

from .config import Config, get_config
from .data import (
    BINARY_MAP,
    load_dataset,
    make_splits,
    to_binary,
)
from .explainer import Explainer
from .fine_tuned import FineTunedPredictor
from .metrics import compute_metrics
from .zero_shot import ZeroShotPredictor

__all__ = [
    "Config",
    "get_config",
    "BINARY_MAP",
    "load_dataset",
    "make_splits",
    "to_binary",
    "compute_metrics",
    "ZeroShotPredictor",
    "FineTunedPredictor",
    "Explainer",
]

__version__ = "0.2.0"
