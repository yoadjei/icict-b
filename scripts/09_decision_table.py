"""E7 deployment decision table for the precise analysis set (ridge, spectral and rainfall).

Per target country it collects the transfer error, its decomposition, the label free shift
score and the few shot outcome, then applies this rule:

    safe k     smallest k at which a calibrated model (offset, linear or pooled) has a 5th
               percentile of R2 over draws above 0 (R2 is measured against the mean of the
               evaluated target fields, so this is beating that mean in about 95 % of draws)
    decision   'deploy as is' if the LOCO R2 95 % lower bound is above 0
               'deploy after calibration' if safe k <= 50
               'calibrate with a larger sample' if 50 < safe k <= 200
               'do not deploy a transferred model' if no k up to 200 qualifies
               (the rule covers transferred models only; a model fitted to local fields
               alone is reported separately in r2_k200_local_ridge)

Usage:
    python scripts/09_decision_table.py
Output:
    results/decision_table.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CALIBRATED = ["offset", "linear", "pooled"]
MODEL, FEATURES = "ridge", "spectral"


def safe_k(summary: pd.DataFrame) -> tuple[float, str]:
    ok = summary[summary["strategy"].isin(CALIBRATED) & (summary["r2_p05"] > 0)]
    if ok.empty:
        return np.nan, "none"
    best = ok.sort_values(["k", "rmse_median"]).iloc[0]
    return float(best["k"]), str(best["strategy"])


def decide(loco_r2_lower: float, k: float) -> str:
    if loco_r2_lower > 0:
        return "deploy as is"
    if np.isnan(k):
        return "do not deploy a transferred model"
    return "deploy after calibration" if k <= 50 else "calibrate with a larger sample"


def main() -> None:
    scores = pd.read_csv(RESULTS / "transfer_scores.csv")
    scores = scores[(scores["subset"] == "precise") & (scores["model"] == MODEL) & (scores["features"] == FEATURES)]
    few = pd.read_csv(RESULTS / "few_shot_summary.csv")
    few = few[(few["model"] == MODEL) & (few["features"] == FEATURES)]
    shift = pd.read_csv(RESULTS / "shift_scenarios.csv")
    shift = shift[(shift["subset"] == "precise")]
    shift = shift[shift["n_sources"] == shift.groupby("target")["n_sources"].transform("max")]
    rows = []
    for target, loco in scores[scores["scheme"] == "loco"].groupby("country"):
        loco = loco.iloc[0]
        within = scores[(scores["scheme"] == "within") & (scores["country"] == target)].iloc[0]
        k, strategy = safe_k(few[few["target"] == target])
        k200 = few[(few["target"] == target) & (few["k"] == 200)].set_index("strategy")
        rows.append({"country": target, "n": int(loco["n"]), "loco_r2": loco["r2"], "loco_rmse": loco["rmse"],
                     "loco_bias": loco["bias"], "offset_share": loco["offset_share"], "scale_share": loco["scale_share"],
                     "unexplained_share": loco["unexplained_share"], "within_r2": within["r2"],
                     "within_rmse": within["rmse"], "domain_auc": shift[shift["target"] == target]["auc"].item(),
                     "safe_k": k, "safe_strategy": strategy,
                     "r2_k200_best_calibrated": k200.loc[CALIBRATED, "r2_median"].max(),
                     "r2p05_k200_best_calibrated": k200.loc[CALIBRATED, "r2_p05"].max(),
                     "r2_k200_local_mean": k200.loc["local_mean", "r2_median"],
                     "r2_k200_local_ridge": k200.loc["local", "r2_median"],
                     "r2p05_k200_local_ridge": k200.loc["local", "r2_p05"],
                     "decision": decide(loco["r2_lo"], k)})
    pd.DataFrame(rows).round(6).to_csv(RESULTS / "decision_table.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
