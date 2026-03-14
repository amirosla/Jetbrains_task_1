"""
Sliding-window feature extraction for incident prediction.

For each time step t we look back W steps and extract per-channel statistics
that capture the current level, spread, dynamics, and recent trend of every
metric.  The window is then labelled 1 if any incident occurs in the next H
steps, and 0 otherwise — turning the problem into binary classification.
"""

import numpy as np
from typing import Tuple


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """Extract a fixed-length feature vector from a single window.

    For each metric channel the following 9 statistics are computed:
        mean, std, min, max, range, net_change, deviation_from_mean,
        linear_trend (slope), recent_volatility (std of last W/4 steps).

    Args:
        window: Array of shape (W, n_metrics).

    Returns:
        1-D feature vector of length n_metrics * 9.
    """
    features: list[float] = []
    W, n_metrics = window.shape
    quarter = max(1, W // 4)

    for col in range(n_metrics):
        s = window[:, col]
        x = np.arange(W, dtype=float)
        slope = float(np.polyfit(x, s, 1)[0])

        features.extend([
            float(s.mean()),
            float(s.std()),
            float(s.min()),
            float(s.max()),
            float(s.max() - s.min()),       # range
            float(s[-1] - s[0]),             # net change over window
            float(s[-1] - s.mean()),         # current deviation from mean
            slope,                           # linear trend
            float(s[-quarter:].std()),       # recent volatility
        ])

    return np.array(features, dtype=np.float32)


def create_dataset(
    metrics: np.ndarray,
    labels: np.ndarray,
    W: int = 50,
    H: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a supervised dataset using a sliding window.

    Each sample consists of features extracted from the W time steps ending at
    time t, and a binary target that is 1 if any incident label fires in the
    forecast horizon [t, t+H).

    Args:
        metrics: Float array of shape (T, n_metrics).
        labels:  Binary int array of shape (T,).
        W:       Look-back window length (number of past steps used as input).
        H:       Forecast horizon (number of future steps to check for incidents).

    Returns:
        X: Feature matrix of shape (N, n_features) where N = T - W - H + 1.
        y: Binary label vector of shape (N,).
    """
    T = len(labels)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for t in range(W, T - H + 1):
        window = metrics[t - W : t]                 # shape (W, n_metrics)
        future_labels = labels[t : t + H]           # shape (H,)

        X_list.append(extract_window_features(window))
        y_list.append(int(future_labels.any()))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int8)
