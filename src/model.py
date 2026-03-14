"""
LightGBM-based binary classifier for incident prediction.

LightGBM was chosen for the following reasons:
  - Handles tabular/statistical features well without deep feature engineering.
  - Built-in support for class imbalance via scale_pos_weight.
  - Fast training, low memory footprint, and strong out-of-the-box performance.
  - Feature importances are easy to interpret, which matters for a real
    alerting system where explainability is valued.

The model wraps a LGBMClassifier with a StandardScaler (helpful for gradient
boosting when features have very different scales) and exposes a clean
predict_proba / predict interface so that the caller can sweep thresholds
independently of training.
"""

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler


class IncidentPredictor:
    """Sliding-window incident predictor backed by LightGBM.

    Args:
        W: Look-back window size (kept for metadata / serialisation).
        H: Forecast horizon (kept for metadata / serialisation).
    """

    def __init__(self, W: int = 50, H: int = 10) -> None:
        self.W = W
        self.H = H
        self.scaler = StandardScaler()
        self.model: lgb.LGBMClassifier | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "IncidentPredictor":
        """Fit scaler and LightGBM classifier.

        Class imbalance is handled by setting scale_pos_weight = |neg| / |pos|,
        which is equivalent to up-weighting minority (incident) samples in the
        loss function.  Early stopping on the validation set prevents
        overfitting without requiring a manual n_estimators search.

        Args:
            X_train: Training feature matrix.
            y_train: Training binary labels.
            X_val:   Optional validation features for early stopping.
            y_val:   Optional validation labels for early stopping.

        Returns:
            self (for chaining).
        """
        X_tr = self.scaler.fit_transform(X_train)

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        self.model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            X_v = self.scaler.transform(X_val)
            fit_kwargs["eval_set"] = [(X_v, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ]

        self.model.fit(X_tr, y_train, **fit_kwargs)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of incident in the next H steps.

        Args:
            X: Feature matrix of shape (N, n_features).

        Returns:
            1-D array of incident probabilities, shape (N,).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using a given alert threshold.

        Args:
            X:         Feature matrix.
            threshold: Probability cutoff above which an alert is raised.

        Returns:
            Binary array of shape (N,).
        """
        return (self.predict_proba(X) >= threshold).astype(np.int8)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def feature_importances(self, feature_names: list[str] | None = None) -> dict:
        """Return a sorted dict of feature importances (gain-based).

        Args:
            feature_names: Optional list of names for the features.

        Returns:
            Dict mapping feature name → importance score, sorted descending.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        importances = self.model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(sorted(zip(names, importances), key=lambda x: x[1], reverse=True))
