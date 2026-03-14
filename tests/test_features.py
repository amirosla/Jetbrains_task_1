"""Unit tests for src/features.py."""

import numpy as np
import pytest

from src.features import create_dataset, extract_window_features


# ---------------------------------------------------------------------------
# extract_window_features
# ---------------------------------------------------------------------------

class TestExtractWindowFeatures:
    def test_output_length(self):
        window = np.random.randn(50, 3)
        feats = extract_window_features(window)
        # 9 statistics × 3 channels = 27
        assert feats.shape == (27,)

    def test_single_channel(self):
        window = np.random.randn(20, 1)
        feats = extract_window_features(window)
        assert feats.shape == (9,)

    def test_constant_signal(self):
        """For a constant signal std, range, net_change, slope should be ~0."""
        window = np.ones((30, 1)) * 5.0
        feats = extract_window_features(window)
        # feats layout: mean(0), std(1), min(2), max(3), range(4),
        #               net_change(5), dev_from_mean(6), slope(7), recent_vol(8)
        assert abs(feats[1]) < 1e-5, "std should be ~0 for constant signal"
        assert abs(feats[4]) < 1e-5, "range should be ~0 for constant signal"
        assert abs(feats[5]) < 1e-5, "net_change should be ~0 for constant signal"
        assert abs(feats[7]) < 1e-5, "slope should be ~0 for constant signal"

    def test_mean_value(self):
        window = np.full((20, 1), 3.0)
        feats = extract_window_features(window)
        np.testing.assert_allclose(feats[0], 3.0, atol=1e-5)

    def test_rising_trend_slope(self):
        window = np.arange(50).reshape(50, 1).astype(float)
        feats = extract_window_features(window)
        assert feats[7] > 0, "Slope should be positive for monotonically rising signal"

    def test_dtype(self):
        window = np.random.randn(20, 2)
        feats = extract_window_features(window)
        assert feats.dtype == np.float32


# ---------------------------------------------------------------------------
# create_dataset
# ---------------------------------------------------------------------------

class TestCreateDataset:
    def _make_data(self, T=500, n_metrics=2):
        metrics = np.random.randn(T, n_metrics).astype(np.float32)
        labels = np.zeros(T, dtype=np.int8)
        labels[100:150] = 1
        return metrics, labels

    def test_output_shapes(self):
        metrics, labels = self._make_data(T=500)
        W, H = 50, 10
        X, y = create_dataset(metrics, labels, W=W, H=H)
        expected_n = 500 - W - H + 1
        assert X.shape[0] == expected_n
        assert y.shape[0] == expected_n

    def test_feature_width(self):
        metrics, labels = self._make_data(T=300, n_metrics=3)
        X, y = create_dataset(metrics, labels, W=20, H=5)
        # 9 features × 3 channels = 27
        assert X.shape[1] == 27

    def test_labels_binary(self):
        metrics, labels = self._make_data()
        _, y = create_dataset(metrics, labels, W=50, H=10)
        assert set(np.unique(y)).issubset({0, 1})

    def test_positive_labels_exist(self):
        """Dataset should contain positive labels when incidents are present."""
        metrics, labels = self._make_data()
        _, y = create_dataset(metrics, labels, W=50, H=10)
        assert y.sum() > 0, "Expected some positive labels near incident interval"

    def test_horizon_labelling(self):
        """Sample at t should be positive iff any label fires in [t, t+H)."""
        T, W, H = 200, 20, 10
        metrics = np.zeros((T, 1))
        labels = np.zeros(T, dtype=np.int8)
        # Single incident at step 100
        labels[100] = 1

        X, y = create_dataset(metrics, labels, W=W, H=H)
        # t maps to original time W + t; incident at 100 → t = 100 - W = 80
        # sample at t=80 has future window [80+W, 80+W+H) = [100, 110) → positive
        # sample at t=79 has future window [99, 109) → also positive (100 is in it)
        # sample at t=70 has future window [90, 100) → negative
        assert y[80] == 1
        assert y[70] == 0
