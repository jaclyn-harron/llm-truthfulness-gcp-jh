"""Reproducible evaluation entrypoint.

    python -m truthfulness.evaluate --data data.csv

Loads a fixed held-out split, runs the zero-shot and fine-tuned predictors over
the *same* test set, and prints the chosen metrics side by side. A fixed seed
makes the split (and therefore the comparison) stable across runs.

If no fine-tuned model is configured (TRUTHFULNESS_FINE_TUNED_MODEL unset), the
script launches a Vertex AI supervised tuning job on the training split first
(unless --skip-finetune is passed) and waits for it — expect 1-3 hours. Use
--sample N to evaluate on a random N-row subset of the held-out set for a quick,
cheaper run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_config
from .data import labels_from_frame, load_dataset, make_splits, points_from_frame
from .explainer import Explainer
from .fine_tuned import FineTunedPredictor
from .metrics import format_metrics_table
from .zero_shot import ZeroShotPredictor


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate truthfulness predictors.")
    parser.add_argument("--data", required=True, help="Path to data.csv")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Evaluate on a random N-row subset of the held-out set (cheaper).",
    )
    parser.add_argument(
        "--skip-finetune", action="store_true",
        help="Do not launch fine-tuning; require a preconfigured fine-tuned model.",
    )
    parser.add_argument(
        "--no-explain", action="store_true",
        help="Skip the explainer demonstration step.",
    )
    parser.add_argument(
        "--explain-examples", type=int, default=3,
        help="How many example explanations to print.",
    )
    parser.add_argument("--json-out", default=None, help="Write metrics JSON to this path.")
    args = parser.parse_args(argv)

    cfg = get_config()
    cfg.require_project()

    df = load_dataset(args.data)
    splits = make_splits(
        df, seed=cfg.seed, test_size=cfg.test_size, val_size=cfg.val_size
    )
    _print_header("Dataset & splits")
    print(f"Loaded {len(df)} de-duplicated rows from {args.data}")
    print(splits.describe())

    test_df = splits.test
    if args.sample is not None and args.sample < len(test_df):
        test_df = test_df.sample(n=args.sample, random_state=cfg.seed).reset_index(drop=True)
        print(f"\nUsing a random sample of {len(test_df)} held-out rows (seed={cfg.seed}).")

    test_points = points_from_frame(test_df)
    test_labels = labels_from_frame(test_df)

    # ---- Zero-shot ----
    _print_header("Zero-shot predictor")
    zs = ZeroShotPredictor(cfg)
    print(f"Model: {zs.model}")
    zs_out = zs.predict(test_points, labels=test_labels)
    print(json.dumps(zs_out.metrics, indent=2))


    # ---- Fine-tuned (Vertex AI supervised tuning) ----
    ft = FineTunedPredictor(cfg)
    if not ft.model and not args.skip_finetune:
        _print_header("Fine-tuning (no tuned model configured)")
        print(f"Launching Vertex AI supervised tuning of {cfg.fine_tune_base_model}...")
        print("This typically takes 1-3 hours; progress is visible in the GCP "
              "console under Vertex AI -> Tuning.")
        res = ft.fine_tune(splits.train)
        print(f"Trained: {res}")
        print(f"\nTuned endpoint: {res.model}")
        print("Pin it for future runs:  export "
              f"TRUTHFULNESS_FINE_TUNED_MODEL={res.model}")
    if not ft.model:
        print(
            "\nNo fine-tuned model available; set TRUTHFULNESS_FINE_TUNED_MODEL "
            "or drop --skip-finetune.",
            file=sys.stderr,
        )
        return 2

    _print_header("Fine-tuned predictor (Vertex AI supervised tuning)")
    print(f"Model: {ft.model}")
    ft_out = ft.predict(test_points, labels=test_labels)
    print(json.dumps(ft_out.metrics, indent=2))


    # ---- Side by side ----
    _print_header("Side-by-side comparison (same held-out split)")
    table = format_metrics_table(
        {"zero-shot": zs_out.metrics, "fine-tuned": ft_out.metrics}
    )
    print(table)
    delta = ft_out.metrics["accuracy"] - zs_out.metrics["accuracy"]
    f1d = ft_out.metrics["macro_f1"] - zs_out.metrics["macro_f1"]
    print(f"\nFine-tuned − zero-shot:  accuracy {delta:+.4f}   macro-F1 {f1d:+.4f}")

    # ---- Explainer demo ----
    if not args.no_explain:
        _print_header(f"Explainer (first {args.explain_examples} held-out points)")
        k = min(args.explain_examples, len(test_points))
        exp = Explainer(cfg)
        exp_out = exp.explain(ft, test_points[:k], labels=test_labels[:k])
        for i, e in enumerate(exp_out.explanations):
            print(f"\n[{i}] gold={test_labels[i]}")
            print(f"    statement: {test_points[i]['statement'][:120]}")
            print(f"    {e.text}")
        if exp_out.metrics:
            print("\nExplainer-reported metrics (underlying model):")
            print(json.dumps(exp_out.metrics, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "seed": cfg.seed,
                    "n_test": len(test_df),
                    "zero_shot": zs_out.metrics,
                    "fine_tuned": ft_out.metrics,
                    "zero_shot_model": zs.model,
                    "fine_tuned_model": ft.model,
                    "fine_tune_base_model": cfg.fine_tune_base_model,
                },
                indent=2,
            )
        )
        print(f"\nWrote metrics to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
