"""
Incident Prediction — main entry point.

Pipeline:
  1. Generate synthetic multivariate time series with labeled incidents.
  2. Build a sliding-window dataset (W look-back steps, H forecast horizon).
  3. Split chronologically: 60 % train / 20 % validation / 20 % test.
  4. Run walk-forward cross-validation to get a robust metric estimate.
  5. Train the final LightGBM model on train+val; early stop on val.
  6. Score the held-out test set and compare against the statistical baseline.
  7. Produce a 6-panel evaluation dashboard and save the trained model.

Usage:
    python main.py
    python main.py --W 50 --H 10 --n_samples 10000 --seed 42
    python main.py --W 30 --H 5  --n_samples 20000 --seed 7
"""

import argparse
import json
import os

import joblib
import numpy as np

from src.baseline import StatisticalBaseline
from src.data_generation import generate_synthetic_timeseries
from src.evaluation import build_feature_names, evaluate_and_plot
from src.features import create_dataset
from src.model import IncidentPredictor
from src.walk_forward import walk_forward_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a sliding-window incident predictor."
    )
    parser.add_argument("--W", type=int, default=50,
                        help="Look-back window size (default: 50)")
    parser.add_argument("--H", type=int, default=10,
                        help="Forecast horizon in steps (default: 10)")
    parser.add_argument("--n_samples", type=int, default=10_000,
                        help="Total time-series length (default: 10 000)")
    parser.add_argument("--n_metrics", type=int, default=3,
                        help="Number of metric channels (default: 3)")
    parser.add_argument("--cv_splits", type=int, default=5,
                        help="Number of walk-forward CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory for plots and metrics (default: results/)")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save the trained model (default: models/)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Incident Prediction  |  Sliding-Window + LightGBM")
    print("=" * 60)
    print(f"  W (look-back)    = {args.W}")
    print(f"  H (horizon)      = {args.H}")
    print(f"  Time steps       = {args.n_samples}")
    print(f"  Metrics          = {args.n_metrics}")
    print(f"  CV splits        = {args.cv_splits}")
    print(f"  Seed             = {args.seed}")
    print()

    # ------------------------------------------------------------------
    # 1. Synthetic data
    # ------------------------------------------------------------------
    print("Step 1 — Generating synthetic time series …")
    metrics_df, incident_labels = generate_synthetic_timeseries(
        n_samples=args.n_samples,
        n_metrics=args.n_metrics,
        seed=args.seed,
    )
    metrics = metrics_df.values
    print(
        f"         {args.n_samples} steps  |  {metrics.shape[1]} metrics  "
        f"|  incident rate = {incident_labels.mean():.2%}\n"
    )

    # ------------------------------------------------------------------
    # 2. Sliding-window dataset
    # ------------------------------------------------------------------
    print(f"Step 2 — Building dataset (W={args.W}, H={args.H}) …")
    X, y = create_dataset(metrics, incident_labels, W=args.W, H=args.H)
    print(
        f"         {X.shape[0]} samples  |  {X.shape[1]} features  "
        f"|  positive rate = {y.mean():.2%}\n"
    )

    # ------------------------------------------------------------------
    # 3. Time-based split  (60 / 20 / 20)
    #    No shuffling — temporal order must be preserved to avoid leakage.
    # ------------------------------------------------------------------
    n = len(X)
    train_end = int(0.60 * n)
    val_end   = int(0.80 * n)

    X_train, y_train = X[:train_end],        y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    print("Step 3 — Chronological split:")
    print(f"         Train = {len(X_train)}  |  Val = {len(X_val)}  |  Test = {len(X_test)}\n")

    # ------------------------------------------------------------------
    # 4. Walk-forward cross-validation
    # ------------------------------------------------------------------
    print(f"Step 4 — Walk-forward cross-validation ({args.cv_splits} folds) …")
    cv_results = walk_forward_cv(X, y, n_splits=args.cv_splits)
    print("\n  CV Summary:")
    for metric, stats in cv_results.items():
        print(f"    {metric.upper():<10}  {stats['mean']:.4f} ± {stats['std']:.4f}")
    print()

    # ------------------------------------------------------------------
    # 5. Final model: train on train+val, early stop on val
    # ------------------------------------------------------------------
    print("Step 5 — Training final LightGBM model …")
    predictor = IncidentPredictor(W=args.W, H=args.H)
    predictor.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    n_trees = predictor.model.best_iteration_ or predictor.model.n_estimators
    print(f"         Done  |  trees used = {n_trees}\n")

    # ------------------------------------------------------------------
    # 6. Baseline scoring
    # ------------------------------------------------------------------
    print("Step 6 — Scoring statistical baseline …")
    baseline = StatisticalBaseline(n_metrics=args.n_metrics)
    baseline_scores = baseline.score(X_test)
    print("         Done\n")

    # ------------------------------------------------------------------
    # 7. Evaluation on held-out test set
    # ------------------------------------------------------------------
    print("Step 7 — Evaluating on test set …\n")
    y_scores = predictor.predict_proba(X_test)
    feat_names = build_feature_names(args.n_metrics)
    importances = predictor.feature_importances(feat_names)

    metrics_dict = evaluate_and_plot(
        y_true=y_test,
        y_scores=y_scores,
        baseline_scores=baseline_scores,
        feature_importances=importances,
        output_dir=args.output_dir,
    )

    # Append CV summary to saved metrics
    metrics_dict["cv"] = cv_results

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # ------------------------------------------------------------------
    # 8. Save model
    # ------------------------------------------------------------------
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "incident_predictor.joblib")
    joblib.dump(predictor, model_path)
    print(f"Model saved to {model_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
