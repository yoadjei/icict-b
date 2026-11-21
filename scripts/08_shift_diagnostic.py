"""E6 label free shift diagnostic across deployment scenarios.

A scenario is a target country and a non empty set of source countries drawn from the others.
For each scenario: the cross validated source versus target classifier AUC (features only)
and the transfer error of a ridge model trained on the sources, relative to the target's
within country (location grouped) error.

Usage:
    python scripts/08_shift_diagnostic.py
Outputs:
    results/shift_scenarios.csv      one row per subset and scenario
    results/shift_correlation.csv    Spearman correlation of AUC with the error ratio
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cropshift import cache, datasets, evaluation, models, shift  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "logs" / "shift_cache"  # scenarios per subset and target, so a run resumes


def target_scenarios(subset: str, target: str, names: list[str], X: np.ndarray, y: np.ndarray,
                     country: np.ndarray, loc: np.ndarray, rmse_within: float) -> pd.DataFrame:
    """Shift score and transfer error for every non empty set of source countries."""
    others = [c for c in names if c != target]
    t = country == target
    rows = []
    for size in range(1, len(others) + 1):
        for sources in itertools.combinations(others, size):
            s = np.isin(country, sources)
            sc = evaluation.scores(y[t], models.fit_predict("ridge", X[s], y[s], X[t]))
            rows.append({"subset": subset, "target": target, "sources": "+".join(sources), "n_sources": size,
                         "auc": shift.domain_auc(X[s], X[t], loc[s], loc[t]), "r2": sc["r2"],
                         "rmse": sc["rmse"], "abs_bias": abs(sc["bias"]), "rmse_ratio": sc["rmse"] / rmse_within})
    return pd.DataFrame(rows)


def main() -> None:
    within = pd.read_csv(RESULTS / "transfer_scores.csv")
    within = within[(within["scheme"] == "within") & (within["model"] == "ridge") & (within["features"] == "spectral")]
    frames = []
    for subset in ("precise", "all"):
        df = datasets.load(subset, embeddings=())
        X, y = df[datasets.SPECTRAL].to_numpy(), df[datasets.TARGET].to_numpy()
        country, loc = df["country"].to_numpy(), df["location"].to_numpy()
        names = list(np.unique(country))
        for target in names:
            rmse_within = within[(within["subset"] == subset) & (within["country"] == target)]["rmse"].item()
            frames.append(cache.cached_frame(
                CACHE / f"{subset}_{target}.parquet",
                lambda: target_scenarios(subset, target, names, X, y, country, loc, rmse_within)))
            print(f"{subset} {target} done", flush=True)
    scen = pd.concat(frames, ignore_index=True)
    scen.round(6).to_csv(RESULTS / "shift_scenarios.csv", index=False)
    corr = []
    for subset, g in scen.groupby("subset"):
        for outcome in ("rmse_ratio", "r2", "abs_bias"):
            rho = spearmanr(g["auc"], g[outcome])
            corr.append({"subset": subset, "outcome": outcome, "scenarios": len(g), "spearman": rho.statistic,
                         "p_value": rho.pvalue, "auc_min": g["auc"].min(), "auc_max": g["auc"].max()})
    pd.DataFrame(corr).round(6).to_csv(RESULTS / "shift_correlation.csv", index=False)
    print("wrote results/shift_scenarios.csv and results/shift_correlation.csv")


if __name__ == "__main__":
    main()
