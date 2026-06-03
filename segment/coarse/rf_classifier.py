"""Coarse-grained Random-Forest classifier (Manuscript: Coarse-grained
classifier training).

A Random Forest (200 trees, max_depth 15) maps the local statistical feature
vector ``f_input`` to an unfavorable-geology category, exposing a per-voxel
probability distribution used both for hard pseudo-labels and for the
confidence (soft) labels consumed by the fine stage.

Each geology type is an INDEPENDENT binary problem, so a classifier is trained
per type with ``num_classes=1`` (``0=background``, ``1=<type>``) and
``predict_proba`` yields the 2-column distribution that matches the fine-stage
binary softmax. (The implementation stays generic in ``num_classes``.) Optional
LR / SVM / Naive-Bayes baselines back the classifier-comparison ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _build_estimator(model: str, params: Optional[Dict[str, Any]] = None):
    params = dict(params or {})
    if model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        defaults = dict(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    if model == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        defaults = dict(max_iter=1000, random_state=42)
        defaults.update(params)
        return LogisticRegression(**defaults)
    if model == "svm":
        from sklearn.svm import SVC

        defaults = dict(probability=True, random_state=42)
        defaults.update(params)
        return SVC(**defaults)
    if model == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB

        return GaussianNB(**params)
    raise ValueError(f"Unknown coarse model: {model!r}")


@dataclass
class CoarseClassifier:
    """Wraps a scikit-learn estimator with a fixed class set so
    ``predict_proba`` always returns columns aligned to ``0..num_classes``."""

    model: str = "random_forest"
    num_classes: int = 1  # foreground classes (excludes background)
    params: Optional[Dict[str, Any]] = None
    estimator: Any = None

    def __post_init__(self):
        if self.estimator is None:
            self.estimator = _build_estimator(self.model, self.params)
        self._classes = np.arange(self.num_classes + 1)  # incl. background=0

    # -- training --------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CoarseClassifier":
        """``X`` (N, F) features, ``y`` (N,) integer labels in ``0..num_classes``."""
        # Impute non-finite feature values (border windows) with column means.
        X = self._sanitize(X, fit=True)
        self.estimator.fit(X, y)
        return self

    # -- inference -------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, num_classes+1) probabilities aligned to ``0..num_classes``,
        robust to classes missing from the training fold."""
        X = self._sanitize(X, fit=False)
        proba = self.estimator.predict_proba(X)
        full = np.zeros((X.shape[0], self.num_classes + 1), dtype=np.float32)
        for col, cls in enumerate(self.estimator.classes_):
            full[:, int(cls)] = proba[:, col]
        return full

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int16)

    # -- helpers ---------------------------------------------------------------
    def _sanitize(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if fit:
            self._col_mean = np.nanmean(
                np.where(np.isfinite(X), X, np.nan), axis=0
            )
            self._col_mean = np.where(
                np.isfinite(self._col_mean), self._col_mean, 0.0
            )
        bad = ~np.isfinite(X)
        if bad.any():
            X = X.copy()
            X[bad] = np.take(self._col_mean, np.where(bad)[1])
        return X

    # -- persistence -----------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "num_classes": self.num_classes,
                "estimator": self.estimator,
                "col_mean": getattr(self, "_col_mean", None),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CoarseClassifier":
        import joblib

        blob = joblib.load(path)
        obj = cls(
            model=blob["model"],
            num_classes=blob["num_classes"],
            estimator=blob["estimator"],
        )
        if blob.get("col_mean") is not None:
            obj._col_mean = blob["col_mean"]
        return obj
