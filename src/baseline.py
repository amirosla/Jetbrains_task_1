"""
Statistical baseline for incident prediction.

Uses a z-score computed from per-channel statistics already present in the
feature matrix: z_i = deviation_from_mean_i / (std_i + ε).  The anomaly
score for a window is the maximum z-score across all metric channels.

This is the simplest non-trivial detector: no training required, fully
interpretable, and representative of rule-based alerting systems in
production.  The LightGBM model must clearly outperform it to justify
its added complexity.

Feature layout (9 values per channel, channels packed consecutively):
  0: mean            3: max             6: deviation_from_mean
  1: std             4: range           7: slope
  2: min             5: net_change      8: recent_volatility
"""

import numpy as np


N_FEATURES_PER_CHANNEL = 9
_IDX_STD = 1
_IDX_DEV = 6          # deviation_from_mean = last_value - mean
_IDX_RECENT_VOL = 8   # std of last W/4 steps


class StatisticalBaseline:
    """Threshold-free z-score anomaly scorer.

    Produces a continuous anomaly score for each window so that AUC-based
    metrics are directly comparable with the LightGBM model.  No fitting is
    needed — the score is derived entirely from within-window statistics.
    """

    def __init__(self, n_metrics: int = 3) -> None:
        self.n_metrics = n_metrics

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample anomaly scores.

        Args:
            X: Feature matrix of shape (N, n_metrics * N_FEATURES_PER_CHANNEL).

        Returns:
            1-D array of anomaly scores, shape (N,).  Higher ↔ more anomalous.
        """
        scores = np.zeros(len(X), dtype=np.float32)

        for ch in range(self.n_metrics):
            base = ch * N_FEATURES_PER_CHANNEL
            std = X[:, base + _IDX_STD] + 1e-8
            dev = X[:, base + _IDX_DEV]
            vol = X[:, base + _IDX_RECENT_VOL]

            # Combine current deviation and recent spike intensity
            z = np.abs(dev) / std + vol / std
            scores = np.maximum(scores, z)

        return scores

    def predict(self, X: np.ndarray, threshold: float = 1.5) -> np.ndarray:
        """Binary predictions at a fixed z-score threshold.

        Args:
            X:         Feature matrix.
            threshold: z-score cutoff (default 1.5σ).

        Returns:
            Binary array of shape (N,).
        """
        return (self.score(X) >= threshold).astype(np.int8)
