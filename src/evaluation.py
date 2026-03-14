"""
Evaluation utilities for the incident prediction model.

Produces a comprehensive 2×3 evaluation figure:
  Row 1: ROC curve (model vs baseline), Precision-Recall curve (model vs baseline),
          Calibration curve (reliability diagram)
  Row 2: Threshold sweep (precision / recall / F1 vs alert threshold),
          Lead-time distribution, Feature importance (top 15)

Key design choices explained inline.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_points: int = 300,
) -> tuple[float, float]:
    """Return the threshold that maximises F1 on the provided scores."""
    thresholds = np.linspace(0.01, 0.99, n_points)
    f1s = [
        f1_score(y_true, (y_scores >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def analyze_lead_times(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[int]:
    """Compute advance-warning lead times for each detected incident onset.

    For each transition 0→1 in ``y_true`` (start of a ground-truth alert
    window), we look backwards to find how many consecutive steps the model
    was already predicting 1 before that onset.  A lead time of k means the
    model raised an alert k steps before the labelled window opened, giving
    additional warning beyond the H-step horizon already baked into the labels.

    Args:
        y_true:  Ground-truth binary labels (ordered in time).
        y_pred:  Binary model predictions (ordered in time).

    Returns:
        List of non-negative integer lead times, one per detected onset.
    """
    lead_times: list[int] = []
    n = len(y_true)

    for i in range(1, n):
        if y_true[i] == 1 and y_true[i - 1] == 0:
            lead = 0
            j = i - 1
            while j >= 0 and y_pred[j] == 1:
                lead += 1
                j -= 1
            lead_times.append(lead)

    return lead_times


def build_feature_names(n_metrics: int) -> list[str]:
    """Return human-readable feature names matching the layout in features.py."""
    stat_names = [
        "mean", "std", "min", "max", "range",
        "net_change", "dev_from_mean", "slope", "recent_vol",
    ]
    return [f"m{i}_{s}" for i in range(n_metrics) for s in stat_names]


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_and_plot(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    baseline_scores: np.ndarray | None = None,
    feature_importances: dict | None = None,
    output_dir: str = "results",
) -> dict:
    """Compute metrics, print a classification report, and save evaluation plots.

    Args:
        y_true:               Ground-truth binary labels (test set).
        y_scores:             Model output probabilities (test set).
        baseline_scores:      Optional baseline anomaly scores for comparison.
        feature_importances:  Optional dict {feature_name: importance_score}.
        output_dir:           Directory where the plot PNG is saved.

    Returns:
        Dict with keys: auc_roc, auc_pr, threshold, f1.
    """
    os.makedirs(output_dir, exist_ok=True)

    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr  = average_precision_score(y_true, y_scores)
    opt_threshold, opt_f1 = find_optimal_threshold(y_true, y_scores)
    y_pred = (y_scores >= opt_threshold).astype(int)

    # --- Console report ---
    print(f"{'─' * 50}")
    print(f"  AUC-ROC             : {auc_roc:.4f}")
    print(f"  AUC-PR              : {auc_pr:.4f}")
    print(f"  Optimal threshold   : {opt_threshold:.3f}  (F1 = {opt_f1:.4f})")
    if baseline_scores is not None:
        bl_auc_roc = roc_auc_score(y_true, baseline_scores)
        bl_auc_pr  = average_precision_score(y_true, baseline_scores)
        print(f"  Baseline AUC-ROC    : {bl_auc_roc:.4f}  (Δ = {auc_roc - bl_auc_roc:+.4f})")
        print(f"  Baseline AUC-PR     : {bl_auc_pr:.4f}  (Δ = {auc_pr  - bl_auc_pr:+.4f})")
    print(f"{'─' * 50}")
    print("\nClassification report (at optimal threshold):")
    print(classification_report(y_true, y_pred, target_names=["No Incident", "Incident"]))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion matrix  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # Lead times
    lead_times = analyze_lead_times(y_true, y_pred)
    if lead_times:
        print(f"\nLead time (steps before alert window): "
              f"mean={np.mean(lead_times):.1f}  "
              f"median={np.median(lead_times):.0f}  "
              f"max={max(lead_times)}")

    # -----------------------------------------------------------------------
    # Figure: 2 × 3 comprehensive evaluation
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Incident Prediction — Evaluation Dashboard", fontsize=15, y=1.01)

    # --- (0,0) ROC curve ---
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    ax.plot(fpr, tpr, lw=2, label=f"LightGBM  AUC={auc_roc:.3f}")
    if baseline_scores is not None:
        b_fpr, b_tpr, _ = roc_curve(y_true, baseline_scores)
        ax.plot(b_fpr, b_tpr, lw=1.5, linestyle="--",
                label=f"Baseline  AUC={bl_auc_roc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")

    # --- (0,1) Precision-Recall curve ---
    ax = axes[0, 1]
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    ax.plot(rec, prec, lw=2, label=f"LightGBM  AUC-PR={auc_pr:.3f}")
    if baseline_scores is not None:
        b_prec, b_rec, _ = precision_recall_curve(y_true, baseline_scores)
        ax.plot(b_rec, b_prec, lw=1.5, linestyle="--",
                label=f"Baseline  AUC-PR={bl_auc_pr:.3f}")
    prevalence = y_true.mean()
    ax.axhline(prevalence, color="grey", linestyle=":", lw=1,
               label=f"Prevalence ({prevalence:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(loc="upper right")

    # --- (0,2) Calibration curve ---
    ax = axes[0, 2]
    # Calibration: how well do predicted probabilities match empirical frequencies?
    # A perfectly calibrated model follows the diagonal y=x.
    fraction_pos, mean_pred = calibration_curve(y_true, y_scores, n_bins=10)
    ax.plot(mean_pred, fraction_pos, "s-", lw=2, label="LightGBM")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend()

    # --- (1,0) Threshold sweep ---
    ax = axes[1, 0]
    thresholds = np.linspace(0.01, 0.99, 300)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        p = (y_scores >= t).astype(int)
        precisions.append(precision_score(y_true, p, zero_division=0))
        recalls.append(recall_score(y_true, p, zero_division=0))
        f1s.append(f1_score(y_true, p, zero_division=0))
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls,    label="Recall")
    ax.plot(thresholds, f1s,        label="F1")
    ax.axvline(opt_threshold, color="red", linestyle="--", lw=1.5,
               label=f"Optimal ({opt_threshold:.2f})")
    ax.set_xlabel("Alert Threshold"); ax.set_ylabel("Score")
    ax.set_title("Metrics vs Alert Threshold"); ax.legend()

    # --- (1,1) Lead-time distribution ---
    ax = axes[1, 1]
    if lead_times:
        max_lead = max(lead_times) if lead_times else 0
        bins = range(0, max_lead + 2)
        ax.hist(lead_times, bins=bins, edgecolor="white", color="steelblue")
        ax.axvline(np.mean(lead_times), color="red", linestyle="--",
                   label=f"Mean = {np.mean(lead_times):.1f} steps")
        ax.set_xlabel("Lead Time (steps before alert window)")
        ax.set_ylabel("Count of Incidents")
        ax.set_title("Advance-Warning Lead-Time Distribution")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No detected incidents", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Lead-Time Distribution")

    # --- (1,2) Feature importance ---
    ax = axes[1, 2]
    if feature_importances:
        top_n = 15
        items = list(feature_importances.items())[:top_n]
        names = [k for k, _ in items]
        values = [v for _, v in items]
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color="steelblue", edgecolor="white")
        ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (gain)")
        ax.set_title(f"Top {top_n} Feature Importances")
    else:
        ax.text(0.5, 0.5, "Feature importances not provided",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Feature Importances")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nEvaluation dashboard saved to {plot_path}")

    return {
        "auc_roc":   auc_roc,
        "auc_pr":    auc_pr,
        "threshold": opt_threshold,
        "f1":        opt_f1,
    }
