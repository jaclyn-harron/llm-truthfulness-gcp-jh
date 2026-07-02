"""Dataset loading, target framing, and leakage-free splits.

The source data (LIAR-style PolitiFact statements) carries a six-way ordinal
label. We map it to a binary target by splitting the ordinal scale down the
middle: the three more-truthful grades -> True, the three less-truthful grades
-> False. See the README ("Target framing") for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Six-way ordinal label -> binary truthful target.
# True  = statement is, on balance, truthful.
# False = statement is, on balance, not truthful (the "positive" class for
#         misinformation detection; see metrics.py).
BINARY_MAP: dict[str, bool] = {
    "true": True,
    "mostly-true": True,
    "half-true": True,
    "barely-true": False,
    "false": False,
    "extremely-false": False,
}

# Attributes shown to the model (everything except the label).
FEATURE_COLUMNS = [
    "statement",
    "subjects",
    "speaker_name",
    "speaker_job",
    "speaker_state",
    "speaker_affiliation",
    "statement_context",
]


@dataclass(frozen=True)
class Splits:
    """Train / validation / test frames. `test` is the shared held-out set used
    to compare predictors fairly; fine-tuning only ever sees `train` (+ `val`)."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> str:
        def line(name: str, df: pd.DataFrame) -> str:
            pos = float(df["label_binary"].mean()) if len(df) else float("nan")
            return f"  {name:5s}: n={len(df):5d}  frac_true={pos:.3f}"

        return "\n".join(
            ["Splits:", line("train", self.train), line("val", self.val), line("test", self.test)]
        )


def to_binary(labels: Iterable[str]) -> list[bool]:
    """Map six-way (or already-binary) labels onto the boolean target."""
    out: list[bool] = []
    for raw in labels:
        if isinstance(raw, bool):
            out.append(raw)
            continue
        key = str(raw).strip().lower()
        if key in BINARY_MAP:
            out.append(BINARY_MAP[key])
        elif key in {"true", "false"}:
            out.append(key == "true")
        else:
            raise ValueError(f"Unrecognised label: {raw!r}")
    return out


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load data.csv, normalise columns, add a `label_binary` column.

    Cleaning is deliberately light: column-name casing is normalised, exact
    duplicate statements are dropped (they would otherwise leak across the
    split), and missing metadata is left as NaN so the prompt builder can omit
    those fields.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "label" not in df.columns:
        raise ValueError(f"Expected a 'label' column; got {list(df.columns)}")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(BINARY_MAP)].copy()

    # Drop exact-duplicate statements to avoid the same text crossing the split.
    df = df.drop_duplicates(subset=["statement"]).reset_index(drop=True)

    df["label_binary"] = to_binary(df["label"])
    return df


def make_splits(df: pd.DataFrame, *, seed: int, test_size: float, val_size: float) -> Splits:
    """Stratified train/val/test split with a fixed seed.

    Stratification is on the binary target so all three frames share the same
    class balance. The split is deterministic given `seed`, which is what makes
    the baseline-vs-fine-tuned comparison fair and reproducible.
    """
    from sklearn.model_selection import train_test_split

    rest, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label_binary"],
    )
    # Carve the validation set out of the remaining (non-test) rows.
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        rest,
        test_size=val_fraction,
        random_state=seed,
        stratify=rest["label_binary"],
    )
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def points_from_frame(df: pd.DataFrame) -> list[dict]:
    """Extract feature-only point dicts (no label) from a frame."""
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    records = df[cols].to_dict(orient="records")
    # Replace NaN with None so prompt building can cleanly skip missing fields.
    cleaned: list[dict] = []
    for rec in records:
        cleaned.append({k: (None if _is_missing(v) else v) for k, v in rec.items()})
    return cleaned


def labels_from_frame(df: pd.DataFrame) -> list[str]:
    return df["label"].tolist()


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))
