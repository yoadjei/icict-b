"""E3 label audit: how splitting and label geolocation precision change within country skill.

(a) Splitting: ridge on spectral and rainfall features, 5 fold cross validation inside each
    country with folds that are random, grouped by location, or grouped by 0.1 degree cell.
(b) Precision: inside one source and country, fields with 30 m and 5 km spatial precision are
    compared at equal sample size (the larger group is subsampled, 20 draws), with location
    grouped cross validation.
(c) Every country, source and precision group with at least 150 fields, location grouped.

Usage:
    python scripts/06_label_audit.py
Outputs:
    results/label_audit_splits.csv, results/label_audit_precision.csv, results/label_audit_groups.csv,
    results/label_audit_locations.csv (fields that share coordinates, per subset)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cropshift import datasets, evaluation, models  # noqa: E402

RESULTS = ROOT / "results"
DRAWS = 20
CONTRASTS = [("Tanzania", "OneAcre_survey"), ("Rwanda", "Niche_trial")]


def cv_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, seed: int = 0) -> float:
    pred = np.empty(len(y))
    splitter = KFold(5, shuffle=True, random_state=seed).split(X) if groups is None else GroupKFold(5).split(X, groups=groups)
    for train, test in splitter:
        pred[test] = models.fit_predict("ridge", X[train], y[train], X[test])
    return evaluation.scores(y, pred)["r2"]


def precision_label(km: pd.Series) -> np.ndarray:
    """GROW-Africa precision as text: 0.03 km is 30 m; five compiled trial records carry 1.85 km."""
    return np.select([km <= datasets.PRECISE_KM, km < 5.0], ["30 m", "1.85 km"], "5 km")


def main() -> None:
    df = datasets.load("all", embeddings=())
    df["precision"] = precision_label(df["precision_km"])
    X_all, y_all = df[datasets.SPECTRAL].to_numpy(), df[datasets.TARGET].to_numpy()

    splits = []
    for subset, frame in (("all", df), ("precise", df[df["precision"] == "30 m"])):
        for country, g in frame.groupby("country"):
            X, y = g[datasets.SPECTRAL].to_numpy(), g[datasets.TARGET].to_numpy()
            splits.append({"subset": subset, "country": country, "n": len(g),
                           "random": cv_r2(X, y, None), "location": cv_r2(X, y, g["location"].to_numpy()),
                           "cell_0.1deg": cv_r2(X, y, g["block"].to_numpy())})
    pd.DataFrame(splits).round(6).to_csv(RESULTS / "label_audit_splits.csv", index=False)

    contrasts = []
    rng = np.random.default_rng(0)
    for country, source in CONTRASTS:
        g = df[(df["country"] == country) & (df["source"] == source)]
        n = g["precision"].value_counts().min()
        for precision, h in g.groupby("precision"):
            for draw in range(DRAWS):
                s = h.iloc[rng.choice(len(h), n, replace=False)] if len(h) > n else h
                contrasts.append({"country": country, "source": source, "precision": precision, "n": n, "draw": draw,
                                  "r2": cv_r2(s[datasets.SPECTRAL].to_numpy(), s[datasets.TARGET].to_numpy(),
                                              s["location"].to_numpy())})
                if len(h) == n:
                    break
    pd.DataFrame(contrasts).round(6).to_csv(RESULTS / "label_audit_precision.csv", index=False)

    groups = []
    for (country, source, precision), g in df.groupby(["country", "source", "precision"]):
        if len(g) >= 150:
            groups.append({"country": country, "source": source, "precision": precision, "n": len(g),
                           "locations": g["location"].nunique(), "yield_mean": g[datasets.TARGET].mean(),
                           "r2_location_cv": cv_r2(g[datasets.SPECTRAL].to_numpy(), g[datasets.TARGET].to_numpy(),
                                                   g["location"].to_numpy())})
    pd.DataFrame(groups).round(6).to_csv(RESULTS / "label_audit_groups.csv", index=False)
    location_summary().to_csv(RESULTS / "label_audit_locations.csv", index=False)
    print("wrote results/label_audit_*.csv")


def location_summary() -> pd.DataFrame:
    """Fields, distinct coordinates and fields that share coordinates with another field, per subset."""
    rows = []
    for subset in ("all", "precise"):
        df = datasets.load(subset, embeddings=())
        rows.append({"subset": subset, "fields": len(df), "locations": df["location"].nunique(),
                     "fields_sharing_coordinates": int(df.duplicated("location", keep=False).sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
