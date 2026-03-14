"""Unit tests for src/model.py."""

import numpy as np
import pytest

from src.model import IncidentPredictor


def _make_xy(n=300, n_feats=27, pos_rate=0.15, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_feats)).astype(np.float32)
    y = (rng.random(n) < pos_rate).astype(np.int8)
    return X, y


class TestIncidentPredictor:
    def test_fit_predict_shapes(self):
        X, y = _make_xy()
        model = IncidentPredictor()
        model.fit(X, y)
        proba = model.predict_proba(X)
        preds = model.predict(X)
        assert proba.shape == (len(X),)
        assert preds.shape == (len(X),)

    def test_probabilities_in_range(self):
        X, y = _make_xy()
        model = IncidentPredictor().fit(X, y)
        proba = model.predict_proba(X)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_respects_threshold(self):
        X, y = _make_xy()
        model = IncidentPredictor().fit(X, y)
        proba = model.predict_proba(X)

        for t in (0.3, 0.5, 0.8):
            preds = model.predict(X, threshold=t)
            expected = (proba >= t).astype(np.int8)
            np.testing.assert_array_equal(preds, expected)

    def test_early_stopping_with_val(self):
        X, y = _make_xy(n=600)
        X_tr, y_tr = X[:400], y[:400]
        X_val, y_val = X[400:], y[400:]
        model = IncidentPredictor()
        model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        # best_iteration_ should be set and less than n_estimators
        assert model.model is not None

    def test_predict_before_fit_raises(self):
        model = IncidentPredictor()
        X, _ = _make_xy()
        with pytest.raises(RuntimeError):
            model.predict_proba(X)

    def test_feature_importances(self):
        X, y = _make_xy(n_feats=9)
        model = IncidentPredictor().fit(X, y)
        names = [f"f{i}" for i in range(9)]
        imp = model.feature_importances(names)
        assert len(imp) == 9
        # Values should be sorted descending
        values = list(imp.values())
        assert values == sorted(values, reverse=True)

    def test_reproducibility(self):
        X, y = _make_xy()
        p1 = IncidentPredictor().fit(X, y).predict_proba(X)
        p2 = IncidentPredictor().fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(p1, p2)
