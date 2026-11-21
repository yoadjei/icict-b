"""E5 few shot local calibration on the precise analysis set.

For each configuration (model, features) and target country, the transferred model is
trained once on the other countries. Then, for k labelled target fields (drawn by location,
repeated R times), every calibration strategy predicts the remaining target fields.

Each (configuration, target country) unit is saved to logs/few_shot_cache as soon as it finishes,
and units already saved are skipped, so an interrupted run resumes without changing the results
(every unit seeds its own random draws).

Usage:
    python scripts/07_few_shot.py --workers 4               # every configuration
    python scripts/07_few_shot.py --configs ridge:spectral  # selected ones; outputs once all are cached
Outputs:
    results/few_shot_draws.csv     one row per configuration, target, k, draw and strategy
    results/few_shot_summary.csv   median and 5th to 95th percentile over draws
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cropshift import calibration, datasets, evaluation, models  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "logs" / "few_shot_cache"  # one file per configuration and target, so a run resumes
KS = [10, 25, 50, 100, 200]
CONFIGS = [("ridge", "spectral", 50), ("hgb", "spectral", 20), ("ridge", "prithvi", 50)]


def target_draws(model: str, feats: str, draws: int, target: str) -> pd.DataFrame:
    """Every draw and strategy for one configuration and target country."""
    df = datasets.load("precise", embeddings=("prithvi",))
    y, country, loc = df[datasets.TARGET].to_numpy(), df["country"].to_numpy(), df["location"].to_numpy()
    X = df[datasets.features(df, feats)].to_numpy()
    src, tgt = np.flatnonzero(country != target), np.flatnonzero(country == target)
    base_tgt = models.make(model).fit(X[src], y[src]).predict(X[tgt])
    target_index = int(np.flatnonzero(np.unique(country) == target)[0])
    rows = []
    for k in KS:
        rng = np.random.default_rng([KS.index(k), target_index])
        for draw in range(draws):
            cal, ev = calibration.draw(loc[tgt], k, rng)
            for strategy in calibration.STRATEGIES:
                p = calibration.predict(strategy, model, base_tgt[cal], base_tgt[ev], y[tgt][cal],
                                        X[src], y[src], X[tgt][cal], X[tgt][ev])
                s = evaluation.scores(y[tgt][ev], p)
                rows.append({"model": model, "features": feats, "target": target, "k": k, "draw": draw,
                             "strategy": strategy, "n_eval": len(ev), "r2": s["r2"], "rmse": s["rmse"],
                             "bias": s["bias"]})
    return pd.DataFrame(rows)


def target_cache(model: str, feats: str, target: str) -> Path:
    return CACHE / f"{model}_{feats}_{target}.csv"


def run_unit(unit: tuple[str, str, int, str]) -> str:
    """Compute one configuration and target and save it atomically, so a rerun skips it."""
    model, feats, draws, target = unit
    with threadpool_limits(1):
        frame = target_draws(model, feats, draws, target)
    path = target_cache(model, feats, target)
    frame.to_csv(path.with_name(path.name + ".tmp"), index=False)
    os.replace(path.with_name(path.name + ".tmp"), path)
    return f"{model} {feats} {target} done"


def config_frame(model: str, feats: str, targets: list[str]) -> pd.DataFrame | None:
    """All draws of one configuration from the cache, or None while a target is missing."""
    whole = CACHE / f"{model}_{feats}.csv"  # written by earlier runs of this script, one file per configuration
    if whole.exists():
        return pd.read_csv(whole)
    parts = [target_cache(model, feats, t) for t in targets]
    if not all(p.exists() for p in parts):
        return None
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=[f"{m}:{f}" for m, f, _ in CONFIGS])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    targets = sorted(datasets.load("precise", embeddings=())["country"].unique())
    CACHE.mkdir(parents=True, exist_ok=True)
    units = [(m, f, d, t) for m, f, d in CONFIGS if f"{m}:{f}" in args.configs
             and not (CACHE / f"{m}_{f}.csv").exists() for t in targets if not target_cache(m, f, t).exists()]
    if units:
        with Pool(min(args.workers, len(units))) as pool:
            for message in pool.imap_unordered(run_unit, units):
                print(message, flush=True)
    frames = [config_frame(m, f, targets) for m, f, _ in CONFIGS]
    if any(frame is None for frame in frames):
        done = sum(frame is not None for frame in frames)
        print(f"{done} of {len(CONFIGS)} configurations complete; outputs are written once all are", flush=True)
        return
    draws_df = pd.concat(frames, ignore_index=True)
    draws_df.round(6).to_csv(RESULTS / "few_shot_draws.csv", index=False)
    summary = (draws_df.groupby(["model", "features", "target", "k", "strategy"])
               .agg(r2_median=("r2", "median"), r2_p05=("r2", lambda v: v.quantile(0.05)),
                    r2_p95=("r2", lambda v: v.quantile(0.95)), rmse_median=("rmse", "median"),
                    rmse_p05=("rmse", lambda v: v.quantile(0.05)), rmse_p95=("rmse", lambda v: v.quantile(0.95)),
                    abs_bias_median=("bias", lambda v: v.abs().median()), draws=("r2", "size"))
               .reset_index())
    summary.round(6).to_csv(RESULTS / "few_shot_summary.csv", index=False)
    print("wrote results/few_shot_draws.csv and results/few_shot_summary.csv")


if __name__ == "__main__":
    main()
