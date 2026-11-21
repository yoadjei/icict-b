"""Few shot local calibration of a transferred model with k labelled target fields.

Calibration fields are drawn by location: whole locations are taken in random order until
k fields are reached (the last location is truncated) and every location that contributed a
calibration field is removed from the evaluation set, so no coordinate is shared.

Strategies
    none         transferred model as is
    offset       transferred model plus the mean residual on the k fields
    linear       a + b * transferred prediction, least squares on the k fields
    pooled       model refitted on the source countries plus the k fields
    local        model fitted on the k fields only
    local_mean   the mean yield of the k fields for every field
"""
from __future__ import annotations

import numpy as np

from cropshift import models

STRATEGIES = ["none", "offset", "linear", "pooled", "local", "local_mean"]


def draw(locations: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Indices of k calibration fields and of the evaluation fields (disjoint locations)."""
    order = rng.permutation(np.unique(locations))
    chosen, used = [], []
    for loc in order:
        if len(chosen) >= k:
            break
        members = rng.permutation(np.flatnonzero(locations == loc))
        chosen.extend(members[:k - len(chosen)])
        used.append(loc)
    chosen = np.array(chosen, dtype=int)
    evaluation = np.flatnonzero(~np.isin(locations, used))
    return chosen, evaluation


def predict(strategy: str, model: str, base_cal: np.ndarray, base_eval: np.ndarray, y_cal: np.ndarray,
            X_source: np.ndarray, y_source: np.ndarray, X_cal: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    """Predictions for the evaluation fields. base_* are the transferred model's predictions."""
    if strategy == "none":
        return base_eval
    if strategy == "offset":
        return base_eval + np.mean(y_cal - base_cal)
    if strategy == "linear":
        b, a = np.polyfit(base_cal, y_cal, 1)
        return a + b * base_eval
    if strategy == "pooled":
        return models.fit_predict(model, np.vstack([X_source, X_cal]), np.r_[y_source, y_cal], X_eval)
    if strategy == "local":
        return models.fit_predict("ridge", X_cal, y_cal, X_eval)
    if strategy == "local_mean":
        return np.full(len(X_eval), np.mean(y_cal))
    raise ValueError(strategy)
