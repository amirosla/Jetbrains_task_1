"""
Incident Prediction — main entry point.

Usage:
    python main.py
    python main.py --W 50 --H 10 --n_samples 10000
    python main.py --W 30 --H 5  --n_samples 20000 --seed 7
"""

import argparse

import numpy as np

from src.data_generation import generate_synthetic_timeseries
from src.evaluation import evaluate_and_plot
from src.features import create_dataset
from src.model import IncidentPredictor


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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory for evaluation plots (default: results/)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 55)
    print("  Incident Prediction  |  Sliding-Window + LightGBM")
    print("=" * 55)
    print(f"  W (look-back)   = {args.W}")
    print(f"  H (horizon)     = {args.H}")
    print(f"  Time steps      = {args.n_samples}")
    print(f"  Seed            = {args.seed}")
    print()

    # ------------------------------------------------------------------
    # 1. Synthetic data
    # ------------------------------------------------------------------
    print("Step 1 — Generating synthetic time series …")
    metrics_df, incident_labels = generate_synthetic_timeseries(
        n_samples=args.n_samples,
        seed=args.seed,
    )
    metrics = metrics_df.values
    print(f"         {args.n_samples} steps  |  {metrics.shape[1]} metrics  "
          f"|  incident rate = {incident_labels.mean():.2%}\n")

    # ------------------------------------------------------------------
    # 2. Sliding-window dataset
    # ------------------------------------------------------------------
    print(f"Step 2 — Building dataset (W={args.W}, H={args.H}) …")
    X, y = create_dataset(metrics, incident_labels, W=args.W, H=args.H)
    print(f"         {X.shape[0]} samples  |  {X.shape[1]} features  "
          f"|  positive rate = {y.mean():.2%}\n")

    # ------------------------------------------------------------------
    # 3. Time-based train / validation / test split  (60 / 20 / 20)
    #    Crucially: NO shuffling — temporal order must be preserved to avoid
    #    data leakage from future observations into the training set.
    # ------------------------------------------------------------------
    n = len(X)
    train_end = int(0.60 * n)
    val_end   = int(0.80 * n)

    X_train, y_train = X[:train_end],       y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],          y[val_end:]

    print(f"Step 3 — Time-based split:")
    print(f"         Train = {len(X_train)}  |  Val = {len(X_val)}  |  Test = {len(X_test)}\n")

    # ------------------------------------------------------------------
    # 4. Model training
    # ------------------------------------------------------------------
    print("Step 4 — Training LightGBM model …")
    predictor = IncidentPredictor(W=args.W, H=args.H)
    predictor.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    n_trees = predictor.model.best_iteration_ or predictor.model.n_estimators
    print(f"         Training complete  |  trees = {n_trees}\n")

    # ------------------------------------------------------------------
    # 5. Evaluation on held-out test set
    # ------------------------------------------------------------------
    print("Step 5 — Evaluating on test set …\n")
    y_scores = predictor.predict_proba(X_test)
    evaluate_and_plot(y_test, y_scores, output_dir=args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
