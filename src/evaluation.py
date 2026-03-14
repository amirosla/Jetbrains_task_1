"""
Evaluation utilities for the incident prediction model.

Three complementary views are produced:
  1. ROC curve  — overall discriminative power (AUC-ROC).
  2. Precision-Recall curve — performance under class imbalance (AUC-PR).
  3. Threshold sweep — how precision, recall, and F1 trade off as the alert
     threshold varies, with the F1-optimal threshold highlighted.

In a real alerting system the threshold is the primary operational dial:
  - Lower threshold → fewer missed incidents, more false alarms (noisy pager).
  - Higher threshold → fewer false alarms, more missed incidents (silent pager).
The threshold sweep plot makes this trade-off explicit so operators can choose
the operating point that fits their SLA.
"""

import os

import matplotlib
matplotlib.use("Agg")          # headless rendering — no display needed
import matplotlib.pyplot as plt
import numpy as np
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
    """Return the threshold that maximises F1 on the provided scores.

    Args:
        y_true:   Ground-truth binary labels.
        y_scores: Predicted probabilities.
        n_points: Number of candidate thresholds to evaluate.

    Returns:
        (best_threshold, best_f1)
    """
    thresholds = np.linspace(0.01, 0.99, n_points)
    f1s = [
        f1_score(y_true, (y_scores >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_and_plot(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    output_dir: str = "results",
) -> dict:
    """Compute metrics, print a report, and save evaluation plots.

    Args:
        y_true:     Ground-truth binary labels (test set).
        y_scores:   Model output probabilities (test set).
        output_dir: Directory where the plot PNG is saved.

    Returns:
        Dict with keys: auc_roc, auc_pr, threshold, f1.
    """
    os.makedirs(output_dir, exist_ok=True)

    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    opt_threshold, opt_f1 = find_optimal_threshold(y_true, y_scores)

    y_pred = (y_scores >= opt_threshold).astype(int)

    print(f"{'─' * 45}")
    print(f"  AUC-ROC            : {auc_roc:.4f}")
    print(f"  AUC-PR             : {auc_pr:.4f}")
    print(f"  Optimal threshold  : {opt_threshold:.3f}  (F1 = {opt_f1:.4f})")
    print(f"{'─' * 45}")
    print("\nClassification report (at optimal threshold):")
    print(classification_report(y_true, y_pred, target_names=["No Incident", "Incident"]))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion matrix  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Incident Prediction — Evaluation", fontsize=14)

    # 1. ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC-ROC = {auc_roc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    # 2. Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    axes[1].plot(rec, prec, lw=2, label=f"AUC-PR = {auc_pr:.3f}")
    axes[1].axhline(y_true.mean(), color="grey", linestyle="--", lw=1,
                    label=f"Baseline (prevalence = {y_true.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    # 3. Threshold sweep
    thresholds = np.linspace(0.01, 0.99, 300)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        p = (y_scores >= t).astype(int)
        precisions.append(precision_score(y_true, p, zero_division=0))
        recalls.append(recall_score(y_true, p, zero_division=0))
        f1s.append(f1_score(y_true, p, zero_division=0))

    axes[2].plot(thresholds, precisions, label="Precision")
    axes[2].plot(thresholds, recalls, label="Recall")
    axes[2].plot(thresholds, f1s, label="F1")
    axes[2].axvline(opt_threshold, color="red", linestyle="--", lw=1.5,
                    label=f"Optimal ({opt_threshold:.2f})")
    axes[2].set_xlabel("Alert Threshold")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Metrics vs Alert Threshold")
    axes[2].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "evaluation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nEvaluation plots saved to {plot_path}")

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "threshold": opt_threshold,
        "f1": opt_f1,
    }
