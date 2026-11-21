"""Regression models: ridge (linear) and histogram gradient boosting (non linear).

OpenMP is limited to one thread: multi threaded histogram gradient boosting stalled on the
Windows and Python 3.14 machine used for the paper, and single threaded fits are deterministic.
"""
from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

threadpool_limits(limits=1, user_api="openmp")

MODELS = ["ridge", "hgb"]
RIDGE_ALPHAS = np.logspace(-2, 5, 29)


def make(name: str):
    """Unfitted estimator. Ridge chooses its penalty by efficient leave one out on the training data."""
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    if name == "hgb":
        return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                             min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    raise ValueError(name)


def fit_predict(name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    return make(name).fit(X_train, y_train).predict(X_test)
