"""
Synthetic time-series generation with labeled incident intervals.

Incidents are simulated as periods of elevated signal mean and increased
variance, mimicking real-world anomalies such as CPU spikes or latency bursts.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_synthetic_timeseries(
    n_samples: int = 10_000,
    n_metrics: int = 3,
    incident_duration: int = 50,
    incident_rate: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate a multivariate time series with labeled incident intervals.

    Each metric is a superposition of two sine waves at different frequencies
    plus Gaussian noise, providing a realistic periodic baseline.  During
    incidents a subset of metrics receives an additive spike drawn from a
    half-normal distribution, simulating a sudden resource exhaustion or
    error-rate surge.

    Args:
        n_samples:        Total number of time steps.
        n_metrics:        Number of independent metric channels.
        incident_duration: Length of each incident window (time steps).
        incident_rate:    Fraction of time steps that belong to an incident.
        seed:             Random seed for reproducibility.

    Returns:
        metrics_df:  DataFrame of shape (n_samples, n_metrics).
        labels:      Binary ndarray of shape (n_samples,); 1 inside incidents.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    data: dict = {}

    for i in range(n_metrics):
        freq1 = rng.uniform(0.005, 0.02)
        freq2 = rng.uniform(0.05, 0.10)
        amplitude = rng.uniform(1.0, 3.0)
        noise_std = rng.uniform(0.1, 0.5)

        signal = (
            amplitude * np.sin(2 * np.pi * freq1 * t)
            + 0.5 * amplitude * np.sin(2 * np.pi * freq2 * t)
            + noise_std * rng.standard_normal(n_samples)
        )
        data[f"metric_{i}"] = signal

    metrics_df = pd.DataFrame(data)

    # --- Incident placement ---
    n_incidents = max(1, int(n_samples * incident_rate / incident_duration))
    labels = np.zeros(n_samples, dtype=np.int8)
    incident_starts: list[int] = []

    margin = min(100, n_samples // 10)
    for _ in range(n_incidents):
        for _ in range(200):  # max placement attempts
            low  = margin
            high = n_samples - incident_duration - margin
            if high <= low:
                break
            start = int(rng.integers(low, high))
            no_overlap = all(
                abs(start - s) >= incident_duration * 1.5
                for s in incident_starts
            )
            if no_overlap:
                incident_starts.append(start)
                break

    for start in incident_starts:
        end = start + incident_duration
        labels[start:end] = 1

        # Inject anomalous signal into a random subset of metrics
        for i in range(n_metrics):
            if rng.random() > 0.3:
                spike = rng.uniform(3.0, 6.0)
                noise = rng.standard_normal(incident_duration).clip(0)
                metrics_df.iloc[start:end, i] += spike * noise * 2

    return metrics_df, labels
