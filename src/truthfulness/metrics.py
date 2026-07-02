"""Shared performance metrics.

Positive class convention
-------------------------
We treat **False** (an untruthful statement) as the positive class. The product
goal is catching misinformation, so recall/precision are reported with respect
to detecting falsehoods. `accuracy` and `macro_f1` are label-symmetric and so do
not depend on this convention; `precision`, `recall`, and `f1` do.

When prediction scores are available (probability that the statement is False),
we also report ROC-AUC and the Brier score (a calibration-sensitive measure).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .data import to_binary


def compute_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    y_score_false: Sequence[float] | None = None,
) -> dict:
    """Compute a metric bundle.

    Parameters
    ----------
    y_true, y_pred : labels (six-way strings or booleans). Mapped to binary
        internally so callers can pass raw labels.
    y_score_false : optional probability that each item is False (positive
        class). Enables ROC-AUC and Brier score.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    yt = np.array(to_binary(y_true))
    yp = np.array(to_binary(y_pred))
    if len(yt) != len(yp):
        raise ValueError(f"length mismatch: {len(yt)} true vs {len(yp)} pred")
    n = len(yt)
    if n == 0:
        return {"n": 0}

    # Positive class = False (untruthful) -> encode as 1.
    pos_true = (~yt).astype(int)
    pos_pred = (~yp).astype(int)

    tn, fp, fn, tp = confusion_matrix(pos_true, pos_pred, labels=[0, 1]).ravel()

    metrics = {
        "n": int(n),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        # w.r.t. positive class = False
        "precision_false": float(precision_score(pos_true, pos_pred, zero_division=0)),
        "recall_false": float(recall_score(pos_true, pos_pred, zero_division=0)),
        "f1_false": float(f1_score(pos_true, pos_pred, zero_division=0)),
        "confusion": {
            "tp_false": int(tp),
            "fp_false": int(fp),
            "tn_true": int(tn),
            "fn_false": int(fn),
        },
        "frac_true_actual": float(yt.mean()),
        "frac_true_pred": float(yp.mean()),
    }

    if y_score_false is not None:
        s = np.asarray(y_score_false, dtype=float)
        if len(s) == n and len(np.unique(pos_true)) == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(pos_true, s))
            except ValueError:
                metrics["roc_auc"] = None
            metrics["brier"] = float(np.mean((s - pos_true) ** 2))
    return metrics


def format_metrics_table(name_to_metrics: dict[str, dict]) -> str:
    """Render a side-by-side comparison table for the eval entrypoint."""
    keys = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced acc."),
        ("macro_f1", "Macro F1"),
        ("f1_false", "F1 (False)"),
        ("precision_false", "Precision (False)"),
        ("recall_false", "Recall (False)"),
        ("roc_auc", "ROC-AUC"),
        ("brier", "Brier (lower=better)"),
    ]
    names = list(name_to_metrics)
    col_w = max(20, *(len(n) for n in names)) + 2
    header = f"{'Metric':22s}" + "".join(f"{n:>{col_w}s}" for n in names)
    lines = [header, "-" * len(header)]
    for key, label in keys:
        row = f"{label:22s}"
        for n in names:
            v = name_to_metrics[n].get(key)
            cell = "    -" if v is None else f"{v:.4f}"
            row += f"{cell:>{col_w}s}"
        lines.append(row)
    # n row
    row = f"{'n':22s}" + "".join(
        f"{str(name_to_metrics[n].get('n', '-')):>{col_w}s}" for n in names
    )
    lines.append(row)
    return "\n".join(lines)
