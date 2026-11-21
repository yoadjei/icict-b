"""Label free shift diagnostic: how well a classifier separates source and target fields."""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def domain_auc(X_source: np.ndarray, X_target: np.ndarray, groups_source: np.ndarray,
               groups_target: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    """Cross validated ROC AUC of a logistic source versus target classifier (0.5 means no shift).

    Folds are grouped by location so that duplicated coordinates do not inflate the AUC.
    Uses features only; no yield label is needed.
    """
    X = np.vstack([X_source, X_target])
    d = np.r_[np.zeros(len(X_source)), np.ones(len(X_target))]
    groups = np.r_[groups_source, groups_target + groups_source.max() + 1]
    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))
    scores = np.zeros(len(d))
    for train, test in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, d, groups):
        scores[test] = clf.fit(X[train], d[train]).predict_proba(X[test])[:, 1]
    return float(roc_auc_score(d, scores))
