"""Unit tests for src/data_generation.py."""

import numpy as np
import pytest

from src.data_generation import generate_synthetic_timeseries


def test_output_shapes():
    df, labels = generate_synthetic_timeseries(n_samples=500, n_metrics=2)
    assert df.shape == (500, 2)
    assert labels.shape == (500,)


def test_label_dtype_and_range():
    _, labels = generate_synthetic_timeseries(n_samples=500)
    assert set(np.unique(labels)).issubset({0, 1})


def test_incident_rate_approximate():
    """Incident rate should be in a reasonable range around the requested 5 %."""
    _, labels = generate_synthetic_timeseries(n_samples=5_000, incident_rate=0.05)
    rate = labels.mean()
    assert 0.01 < rate < 0.20, f"Incident rate {rate:.3f} out of expected range"


def test_reproducibility():
    df1, l1 = generate_synthetic_timeseries(n_samples=200, seed=0)
    df2, l2 = generate_synthetic_timeseries(n_samples=200, seed=0)
    np.testing.assert_array_equal(df1.values, df2.values)
    np.testing.assert_array_equal(l1, l2)


def test_different_seeds_differ():
    _, l1 = generate_synthetic_timeseries(n_samples=200, seed=0)
    _, l2 = generate_synthetic_timeseries(n_samples=200, seed=99)
    assert not np.array_equal(l1, l2)


def test_column_names():
    df, _ = generate_synthetic_timeseries(n_samples=100, n_metrics=4)
    assert list(df.columns) == [f"metric_{i}" for i in range(4)]
