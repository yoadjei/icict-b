"""Publication figures from the result tables (vector PDF and 300 dpi PNG in figures/).

Usage:
    python scripts/10_figures.py
Outputs:
    figures/fig1_label_audit    splitting scheme and label precision versus within country R2
    figures/fig2_few_shot       R2 of calibration strategies versus number of local labels
    figures/fig3_shift          label free shift score versus transfer error, precise and full sets
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cropshift import plotting  # noqa: E402
from cropshift.plotting import MARKERS, OKABE_ITO, WIDTH_IN  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
COUNTRIES = ["Kenya", "Malawi", "Rwanda", "Tanzania"]
STRATEGIES = {"none": "transferred as is", "offset": "offset correction", "linear": "linear recalibration",
              "pooled": "pooled retraining", "local": "local model only", "local_mean": "local mean"}


def fig_label_audit() -> None:
    splits = pd.read_csv(RESULTS / "label_audit_splits.csv")
    splits = splits[splits["subset"] == "precise"].set_index("country").loc[COUNTRIES]
    prec = pd.read_csv(RESULTS / "label_audit_precision.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH_IN, 1.9), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(COUNTRIES))
    for i, (col, label) in enumerate([("random", "random folds"), ("location", "grouped by location"),
                                      ("cell_0.1deg", "grouped by 0.1° cell")]):
        a.plot(x + (i - 1) * 0.18, splits[col], MARKERS[i], color=OKABE_ITO[i], label=label, linestyle="none")
    a.axhline(0, color="0.5", lw=0.6, ls=":")
    a.set_xticks(x, COUNTRIES)
    a.set_ylabel("within country $R^2$")
    a.set_title("(a)", loc="left")
    a.legend(loc="lower left", handletextpad=0.2)
    groups = [("Tanzania", "OneAcre_survey", "Tanzania\nsurvey"), ("Rwanda", "Niche_trial", "Rwanda\ntrial")]
    for j, (country, source, label) in enumerate(groups):
        for i, precision in enumerate(["30 m", "5 km"]):
            v = prec[(prec["country"] == country) & (prec["source"] == source) & (prec["precision"] == precision)]["r2"]
            med = v.median()
            lo, hi = v.quantile(0.05), v.quantile(0.95)
            b.errorbar(j + (i - 0.5) * 0.3, med, yerr=[[med - lo], [hi - med]], fmt=MARKERS[i], color=OKABE_ITO[i + 4],
                       capsize=2, lw=0.8, label=f"{precision} precision" if j == 0 else None)
    b.axhline(0, color="0.5", lw=0.6, ls=":")
    b.set_xticks([0, 1], [g[2] for g in groups])
    b.set_xlim(-0.6, 1.6)
    b.set_ylabel("$R^2$ at equal sample size")
    b.set_title("(b)", loc="left")
    b.legend(loc="lower left", handletextpad=0.2)
    fig.tight_layout(w_pad=1.0)
    plotting.save(fig, FIGURES / "fig1_label_audit")


def fig_few_shot(model: str = "ridge", features: str = "spectral") -> None:
    few = pd.read_csv(RESULTS / "few_shot_summary.csv")
    few = few[(few["model"] == model) & (few["features"] == features)]
    scores = pd.read_csv(RESULTS / "transfer_scores.csv")
    within = scores[(scores["subset"] == "precise") & (scores["model"] == model) & (scores["features"] == features)
                    & (scores["scheme"] == "within")].set_index("country")["r2"]
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH_IN, 1.75), sharey=True)
    for ax, country in zip(axes, COUNTRIES):
        g = few[few["target"] == country]
        for i, (strategy, label) in enumerate(STRATEGIES.items()):
            s = g[g["strategy"] == strategy].sort_values("k")
            style = dict(color="0.45", ls="--", marker=None) if strategy == "none" else dict(color=OKABE_ITO[i - 1], marker=MARKERS[i - 1])
            ax.plot(s["k"], s["r2_median"], label=label, **style)
            if strategy == "offset":
                ax.fill_between(s["k"], s["r2_p05"], s["r2_p95"], color=OKABE_ITO[i - 1], alpha=0.15, lw=0)
        ax.axhline(within[country], color="0.2", lw=0.7, ls=":")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100, 200], ["10", "25", "50", "100", "200"])
        ax.minorticks_off()
        ax.set_title(country)
        ax.set_xlabel("local labels $k$")
    axes[0].set_ylabel("$R^2$ on remaining fields")
    # keep the uncalibrated model and the whole offset band in view
    lowest = min(few[few["strategy"] == "none"]["r2_median"].min(), few[few["strategy"] == "offset"]["r2_p05"].min())
    lo = np.floor(lowest * 10) / 10
    axes[0].set_ylim(lo, max(0.25, within.max() + 0.05))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17), columnspacing=1.0)
    fig.tight_layout(w_pad=0.4)
    plotting.save(fig, FIGURES / "fig2_few_shot")


def fig_shift() -> None:
    scen = pd.read_csv(RESULTS / "shift_scenarios.csv")
    corr = pd.read_csv(RESULTS / "shift_correlation.csv").set_index(["subset", "outcome"])
    targets = sorted(scen["target"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.0), sharey=True)
    for ax, (subset, label) in zip(axes, [("precise", "(a) precise set"), ("all", "(b) full set")]):
        g = scen[scen["subset"] == subset]
        for i, country in enumerate(targets):
            h = g[g["target"] == country]
            if len(h):
                ax.plot(h["auc"], h["rmse_ratio"], MARKERS[i], color=OKABE_ITO[i], label=country, linestyle="none", ms=3.5)
        ax.axhline(1, color="0.5", lw=0.6, ls=":")
        ax.set_xlabel("source versus target AUC")
        rho = corr.loc[(subset, "rmse_ratio")]
        ax.set_title(f"{label}, $n$ = {int(rho['scenarios'])}, $\\rho$ = {rho['spearman']:.2f}", loc="left")
    axes[0].set_ylabel("transfer RMSE / within RMSE")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(targets), bbox_to_anchor=(0.5, -0.1), columnspacing=1.0,
               handletextpad=0.2)
    fig.tight_layout(w_pad=0.8)
    plotting.save(fig, FIGURES / "fig3_shift")


def main() -> None:
    plotting.apply_style()
    fig_label_audit()
    fig_few_shot()
    fig_shift()
    print("wrote figures/fig1_label_audit, fig2_few_shot, fig3_shift (.pdf, .png)")


if __name__ == "__main__":
    main()
