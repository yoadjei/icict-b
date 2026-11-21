"""Build the tabular feature table from the cached Earth Engine extracts.

Filters, in order, with counts recorded per country:
  1. a cached extract exists for the field;
  2. at least 80% of patch pixels are valid after cloud masking;
  3. yield within 3 SD of its country mean.

Outputs:
    data/processed/master_features.parquet   (gitignored)
    results/dataset_summary.csv              per country yield statistics of the final sample
    results/sample_flow.csv                  per country counts after each filter
    results/season_window_sources.csv        how each field's season window was defined
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cropshift.features import field_features  # noqa: E402

LABELS = ROOT / "data" / "processed" / "labels_sampled.parquet"
FIELDS = ROOT / "data" / "interim" / "fields"
OUT = ROOT / "data" / "processed" / "master_features.parquet"
RESULTS = ROOT / "results"
MIN_VALID_FRACTION = 0.80
YIELD_OUTLIER_SD = 3.0
KEEP_LABEL_COLUMNS = ["field_id", "country", "source", "year", "harvestyear", "agyearstart",
                      "agyearend", "lat", "lon", "spatialprecision_km_", "yield_kgha"]


def main() -> None:
    labels = pd.read_parquet(LABELS)
    flow = {"sampled": labels.groupby("country").size()}

    rows = []
    for _, lab in labels.iterrows():
        path = FIELDS / f"{lab['field_id']}.npz"
        if not path.exists():
            continue
        with np.load(path) as npz:
            rows.append({**lab[KEEP_LABEL_COLUMNS].to_dict(), **field_features(npz)})
    df = pd.DataFrame(rows).rename(columns={"spatialprecision_km_": "precision_km"})
    flow["extracted"] = df.groupby("country").size()

    df = df[df["valid_fraction"] >= MIN_VALID_FRACTION]
    flow["cloud_filter"] = df.groupby("country").size()

    stats = df.groupby("country")["yield_kgha"].agg(["mean", "std"])
    z = (df["yield_kgha"] - df["country"].map(stats["mean"])).abs() / df["country"].map(stats["std"])
    df = df[z <= YIELD_OUTLIER_SD].reset_index(drop=True)
    flow["outlier_filter"] = df.groupby("country").size()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    flow_df = pd.DataFrame(flow).fillna(0).astype(int)
    flow_df.loc["Total"] = flow_df.sum()
    flow_df.to_csv(RESULTS / "sample_flow.csv")

    summary = df.groupby("country")["yield_kgha"].agg(n="size", mean="mean", sd="std", min="min", max="max")
    summary.loc["Total"] = [len(df), df["yield_kgha"].mean(), df["yield_kgha"].std(),
                            df["yield_kgha"].min(), df["yield_kgha"].max()]
    summary.round(1).to_csv(RESULTS / "dataset_summary.csv")
    window_mix = pd.crosstab(df["country"], df["window_source"])
    window_mix.to_csv(RESULTS / "season_window_sources.csv")

    print(flow_df.to_string())
    print(summary.round(0).to_string())
    print(window_mix.to_string())
    print(f"saved {OUT} with {len(df)} rows")


if __name__ == "__main__":
    main()
