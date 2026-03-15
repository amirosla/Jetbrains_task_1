"""
Walk-forward (expanding-window) cross-validation for time-series data.

Standard k-fold CV must not be used on time series: randomly shuffled folds
allow future observations to leak into the training set, producing optimistic
metrics that do not reflect real deployment performance.

Walk-forward CV avoids this by always training on the past and validating on
the immediate future, mimicking the sequential nature of production inference:

  Fold 1:  [====train====|--val--]
  Fold 2:  [=======train========|--val--]
  Fold 3:  [============train===========|--val--]
  ...
                                                → time

Each fold expands the training window by one validation block.  Metrics are
averaged across folds to give a more reliable estimate than a single split.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from src.model import IncidentPredictor
from src.evaluation import find_optimal_threshold


def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    min_train_fraction: float = 0.40,
) -> dict:
    """Run expanding-window cross-validation and return aggregated metrics.

    Args:
        X:                    Feature matrix (temporally ordered).
        y:                    Binary label vector (temporally ordered).
        n_splits:             Number of validation folds.
        min_train_fraction:   Minimum fraction of data used for the first
                              training fold (default 0.40).

    Returns:
        Dict with keys ``auc_roc``, ``auc_pr``, ``f1`` — each a dict with
        ``mean`` and ``std`` across folds.
    """
    n = len(X)
    min_train = int(min_train_fraction * n)
    remaining = n - min_train
    fold_size = remaining // (n_splits + 1)  # +1 so last fold has a test block

    fold_metrics: list[dict] = []

    print(f"  Walk-forward CV: {n_splits} folds, min train = {min_train} samples")

    for fold in range(n_splits):
        train_end = min_train + fold * fold_size
        val_end = train_end + fold_size

        if val_end > n:
            break

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]

        if y_val.sum() == 0:
            print(f"  Fold {fold + 1}/{n_splits}  skipped — no positive labels in validation window")
            continue

        predictor = IncidentPredictor()
        predictor.fit(X_tr, y_tr)
        scores = predictor.predict_proba(X_val)

        auc_roc = roc_auc_score(y_val, scores)
        auc_pr = average_precision_score(y_val, scores)
        threshold, f1 = find_optimal_threshold(y_val, scores)

        fold_metrics.append({"auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1})

        print(
            f"  Fold {fold + 1}/{n_splits}  "
            f"train={train_end}  val={fold_size}  "
            f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  F1={f1:.4f}"
        )

    if not fold_metrics:
        raise ValueError("No valid folds produced — increase n_samples or reduce n_splits.")

    result: dict = {}
    for metric in ("auc_roc", "auc_pr", "f1"):
        values = [fm[metric] for fm in fold_metrics]
        result[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    return result
