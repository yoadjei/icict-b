"""E1 leave one country out (LOCO) transfer, E2 within country reference, E4 error decomposition.

For every analysis set, feature set and model:
    loco      train on all other countries, predict the held out country
    within    5 fold cross validation inside the country, folds grouped by location
A source mean baseline (predict the mean yield of the training countries) is added for LOCO.
Ridge is run on every feature set; gradient boosting on the spectral and rainfall features only.

Predictions of every (subset, features, model, scheme) step are saved to logs/transfer_cache
when computed and reused on rerun, so an interrupted run resumes.

Usage:
    python scripts/05_transfer.py
Outputs:
    results/transfer_predictions.parquet   one row per field, configuration and scheme
    results/transfer_scores.csv            scores, Murphy decomposition, 95 % location bootstrap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cropshift import cache, datasets, evaluation, models  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "logs" / "transfer_cache"  # predictions per subset, features, model and scheme


def loco(X: np.ndarray, y: np.ndarray, country: np.ndarray, model: str) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for target in np.unique(country):
        test = country == target
        pred[test] = models.fit_predict(model, X[~test], y[~test], X[test])
    return pred


def within(X: np.ndarray, y: np.ndarray, country: np.ndarray, groups: np.ndarray, model: str) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for target in np.unique(country):
        idx = np.flatnonzero(country == target)
        for train, test in GroupKFold(5).split(idx, groups=groups[idx]):
            pred[idx[test]] = models.fit_predict(model, X[idx[train]], y[idx[train]], X[idx[test]])
    return pred


def source_mean(y: np.ndarray, country: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y))
    for target in np.unique(country):
        pred[country == target] = y[country != target].mean()
    return pred


def summarise(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = preds.merge(df[["field_id", "country", "location", datasets.TARGET]], on="field_id")
    for keys, g in merged.groupby(["subset", "features", "model", "scheme", "country"]):
        y, p, loc = g[datasets.TARGET].to_numpy(), g["pred"].to_numpy(), g["location"].to_numpy()
        lo, hi = evaluation.bootstrap_interval(y, p, loc, "r2")
        rlo, rhi = evaluation.bootstrap_interval(y, p, loc, "rmse")
        rows.append({**dict(zip(["subset", "features", "model", "scheme", "country"], keys)),
                     **evaluation.scores(y, p), "r2_lo": lo, "r2_hi": hi, "rmse_lo": rlo, "rmse_hi": rhi,
                     "yield_mean": y.mean(), "yield_sd": y.std(), **evaluation.murphy(y, p)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="*", default=["precise", "all"])
    parser.add_argument("--features", nargs="*", default=datasets.FEATURE_SETS)
    parser.add_argument("--models", nargs="*", default=models.MODELS)
    args = parser.parse_args()
    all_preds, all_scores = [], []
    for subset in args.subsets:
        df = datasets.load(subset)
        y, country, loc = df[datasets.TARGET].to_numpy(), df["country"].to_numpy(), df["location"].to_numpy()
        frames = [pd.DataFrame({"field_id": df["field_id"], "subset": subset, "features": "none",
                                "model": "source_mean", "scheme": "loco", "pred": source_mean(y, country)})]
        for feats in args.features:
            X = df[datasets.features(df, feats)].to_numpy()
            for model in args.models:
                if model == "hgb" and feats != "spectral":
                    continue  # gradient boosting is run on the 17 spectral and rainfall features only
                for scheme, predict in (("loco", lambda: loco(X, y, country, model)),
                                        ("within", lambda: within(X, y, country, loc, model))):
                    frames.append(cache.cached_frame(
                        CACHE / f"{subset}_{feats}_{model}_{scheme}.parquet",
                        lambda: pd.DataFrame({"field_id": df["field_id"], "subset": subset, "features": feats,
                                              "model": model, "scheme": scheme, "pred": predict()})))
                print(f"{subset} {feats} {model} done", flush=True)
        preds = pd.concat(frames, ignore_index=True)
        all_preds.append(preds)
        all_scores.append(summarise(df, preds))
    RESULTS.mkdir(exist_ok=True)
    pd.concat(all_preds).to_parquet(RESULTS / "transfer_predictions.parquet", index=False)
    pd.concat(all_scores).round(6).to_csv(RESULTS / "transfer_scores.csv", index=False)
    print("wrote results/transfer_predictions.parquet and results/transfer_scores.csv")


if __name__ == "__main__":
    main()
