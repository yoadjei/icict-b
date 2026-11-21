"""Scores, the Murphy decomposition of the mean squared error, and location bootstrap intervals."""
from __future__ import annotations

import numpy as np


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """R2 is computed against the evaluated set's own mean (0 means no better than that mean)."""
    err = p - y
    mse = float(np.mean(err ** 2))
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else 0.0
    return {"n": len(y), "r2": 1 - mse / float(np.var(y)), "rmse": mse ** 0.5,
            "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err)), "r": r}


def murphy(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """MSE = offset + scale + unexplained (population moments).

    offset       (mean p - mean y)^2            removed by an intercept correction
    scale        (sd p - r sd y)^2              additionally removed by a linear recalibration
    unexplained  sd y^2 (1 - r^2)               reduced only by better ranking of fields
    """
    r = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else 0.0
    sy, sp = float(np.std(y)), float(np.std(p))
    parts = {"offset": (p.mean() - y.mean()) ** 2, "scale": (sp - r * sy) ** 2, "unexplained": sy ** 2 * (1 - r ** 2)}
    total = sum(parts.values())
    return {**{k: float(v) for k, v in parts.items()}, "mse": float(total),
            **{f"{k}_share": float(v / total) for k, v in parts.items()}}


def bootstrap_interval(y: np.ndarray, p: np.ndarray, groups: np.ndarray, metric: str,
                       n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval from resampling locations (clusters) with replacement."""
    rng = np.random.default_rng(seed)
    codes, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(codes))]
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(codes), len(codes))])
        values.append(scores(y[idx], p[idx])[metric])
    tail = (1 - level) / 2 * 100
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))
