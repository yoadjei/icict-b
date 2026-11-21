"""LaTeX tables for the paper, generated from the result files (results/tables/*.tex).

Usage:
    python scripts/11_tables.py
Outputs:
    results/tables/data.tex        fields, label precision, harvest years and yields per country
    results/tables/transfer.tex    within country and leave one country out scores, error shares
    results/tables/features.tex    LOCO R2 by model and feature set
    results/tables/decision.tex    deployment decision table
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cropshift import earth_engine  # noqa: E402

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
COUNTRIES = ["Kenya", "Malawi", "Rwanda", "Tanzania"]
SOURCE_LABELS = {"OneAcre_survey": "OAF", "Niche_trial": "Niche", "LSMS_MWI2019": "LSMS", "LSMS_NGA2018": "LSMS",
                 "Corbeels2020": "Corbeels"}


def num(x: float, digits: int = 3) -> str:
    """Number for a table cell, with a true minus sign."""
    s = f"{x:.{digits}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def thousands(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def write(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(body, encoding="utf-8")


def data_table() -> None:
    df = pd.read_parquet(ROOT / "data" / "processed" / "master_features.parquet")
    df["harvest_year"] = df.apply(earth_engine.harvest_year, axis=1)
    df["precise"] = df["precision_km"] <= 0.03
    lines = [r"\begin{tabular}{@{}lrrllr@{}}", r"\toprule",
             r"Country & Fields & At 30 m & Sources (30 m) & Harvests (30 m) & Yield (30 m), kg/ha \\", r"\midrule"]
    for country, g in df.groupby("country"):
        p = g[g["precise"]]
        if len(p):
            sources = ", ".join(sorted({SOURCE_LABELS[s] for s in p["source"]}))
            years = f"{p['harvest_year'].min()} to {p['harvest_year'].max()}" if p["harvest_year"].nunique() > 1 else str(p["harvest_year"].min())
            yld = f"{thousands(p['yield_kgha'].mean())} ({thousands(p['yield_kgha'].std())})"
        else:
            sources, years, yld = "none", "", ""
        lines.append(f"{country} & {thousands(len(g))} & {thousands(len(p))} & {sources} & {years} & {yld} \\\\")
    p = df[df["precise"]]
    lines += [r"\midrule", f"Total & {thousands(len(df))} & {thousands(len(p))} & & & "
              f"{thousands(p['yield_kgha'].mean())} ({thousands(p['yield_kgha'].std())}) \\\\", r"\bottomrule", r"\end{tabular}"]
    write("data", "\n".join(lines) + "\n")


def transfer_table() -> None:
    s = pd.read_csv(RESULTS / "transfer_scores.csv")
    s = s[(s["subset"] == "precise") & (s["model"] == "ridge") & (s["features"] == "spectral")]
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Within country} & \multicolumn{5}{c}{Leave one country out} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-8}",
             r"Target & $R^2$ & RMSE & $R^2$ [interval] & RMSE & Bias & Offset & Scale \\", r"\midrule"]
    for country in COUNTRIES:
        w = s[(s["scheme"] == "within") & (s["country"] == country)].iloc[0]
        l = s[(s["scheme"] == "loco") & (s["country"] == country)].iloc[0]
        lines.append(f"{country} & {num(w['r2'])} & {thousands(w['rmse'])} & {num(l['r2'])} [{num(l['r2_lo'])}, {num(l['r2_hi'])}] & "
                     f"{thousands(l['rmse'])} & {thousands(l['bias']).replace('-', '$-$')} & "
                     f"{100 * l['offset_share']:.0f}\\% & {100 * l['scale_share']:.0f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("transfer", "\n".join(lines) + "\n")


def features_table() -> None:
    s = pd.read_csv(RESULTS / "transfer_scores.csv")
    s = s[s["subset"] == "precise"]
    configs = [("source_mean", "none", "Source country mean"), ("ridge", "spectral", "Ridge, tabular features"),
               ("hgb", "spectral", "Gradient boosting, tabular features"), ("ridge", "prithvi", "Ridge, Prithvi-EO"),
               ("ridge", "vit", "Ridge, ViT-Base"), ("ridge", "spectral+prithvi", "Ridge, tabular and Prithvi-EO")]
    lines = [r"\begin{tabular}{@{}l" + "r" * (len(COUNTRIES) + 1) + "@{}}", r"\toprule",
             r" & \multicolumn{" + str(len(COUNTRIES)) + r"}{c}{LOCO $R^2$ by target} & Within $R^2$ \\",
             r"\cmidrule(lr){2-" + str(len(COUNTRIES) + 1) + "}",
             "Model and features & " + " & ".join(COUNTRIES) + r" & (mean) \\", r"\midrule"]
    for model, feats, label in configs:
        loco = s[(s["model"] == model) & (s["features"] == feats) & (s["scheme"] == "loco")].set_index("country")
        within = s[(s["model"] == model) & (s["features"] == feats) & (s["scheme"] == "within")]
        if loco.empty:
            continue
        w = num(within["r2"].mean()) if len(within) else ""
        lines.append(f"{label} & " + " & ".join(num(loco.loc[c, "r2"]) for c in COUNTRIES) + f" & {w} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("features", "\n".join(lines) + "\n")


def decision_table() -> None:
    d = pd.read_csv(RESULTS / "decision_table.csv").set_index("country").loc[COUNTRIES]
    # The decision column is shown only when the rule's decision differs between countries;
    # otherwise the paper states the shared decision in the caption.
    show_decision = d["decision"].nunique() > 1
    header = r"Target & LOCO $R^2$ & AUC & Safe $k$ & $R^2_{200}$ calibrated & $R^2_{200}$ local mean"
    lines = [r"\begin{tabular}{@{}lrrrrr" + ("l" if show_decision else "") + "@{}}", r"\toprule",
             header + (r" & Decision \\" if show_decision else r" \\"),
             r"\midrule"]
    for country, r in d.iterrows():
        k = "none" if pd.isna(r["safe_k"]) else f"{int(r['safe_k'])} ({r['safe_strategy']})"
        decision = f" & {r['decision']}" if show_decision else ""
        lines.append(f"{country} & {num(r['loco_r2'])} & {num(r['domain_auc'])} & {k} & {num(r['r2_k200_best_calibrated'])} & "
                     f"{num(r['r2_k200_local_mean'])}{decision} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("decision", "\n".join(lines) + "\n")


def main() -> None:
    data_table()
    transfer_table()
    features_table()
    decision_table()
    print("wrote results/tables/{data,transfer,features,decision}.tex")


if __name__ == "__main__":
    main()
