#!/usr/bin/env python3
"""Phase 1 - Extract and filter GROW-Africa labels from ZIP."""

import numpy as np
import pandas as pd
import zipfile
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_ZIP = RAW_DIR / "GROW-Africa-Database.zip"
OUTPUT_PATH = PROCESSED_DIR / "labels_sampled.parquet"
COUNTS_PATH = PROCESSED_DIR / "sample_counts.csv"
REPORT_PATH = PROCESSED_DIR / "sample_report.csv"

PRIMARY_CROP = "maize"
YEAR_MIN, YEAR_MAX = 2017, 2022
MIN_OBSERVATIONS_PER_COUNTRY = 100
CANDIDATE_COUNTRIES = ["Rwanda", "Kenya", "Tanzania", "Malawi", "Nigeria"]

TARGET_TOTAL = 8_000
N_YIELD_BINS = 5
RANDOM_SEED = 42


def load_grow_africa_from_zip(zip_path):
    """Load GROW-Africa data from ZIP, handling multiple file formats."""
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    print(f"Loading GROW-Africa from {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)...")

    with zipfile.ZipFile(zip_path) as zf:
        all_names = [n for n in zf.namelist() if not n.startswith("__MACOSX")]

        GPS_FILES = ("point", "lsms_cropcut", "lsms_survey")
        data_files = [
            n for n in all_names
            if (n.endswith(".xlsx") or n.endswith(".csv"))
            and any(tag in n.lower() for tag in GPS_FILES)
        ]

        if not data_files:
            raise ValueError(f"No GPS-level data files in ZIP. Contents: {all_names}")

        frames = []
        for entry in data_files:
            print(f"  Reading {entry}...")
            try:
                raw = zf.read(entry)
                if entry.endswith(".xlsx"):
                    import openpyxl
                    frame = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
                else:
                    frame = pd.read_csv(io.BytesIO(raw))
                frame["_source_file"] = entry
                frames.append(frame)
                print(f"    OK {len(frame):,} rows")
            except Exception as e:
                print(f"    WARNING: {e}")

    if not frames:
        raise ValueError("All files in ZIP failed to load.")

    df = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    return df


def normalise_columns(df):
    """Lowercase and coalesce column name aliases."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    aliases = {
        "crop_name": ["crop", "crop_type", "cropname"],
        "yield_kgha": ["yield_ton_ha_", "yield_kg_ha", "yield_kg/ha",
                       "grain_yield", "yield_t_ha", "yieldtonha"],
        "lat": ["latitude", "gps_lat", "y"],
        "lon": ["longitude", "gps_lon", "x"],
        "year": ["harvestyear", "agyearend", "agyearstart",
                 "harvest_year", "survey_year", "crop_year"],
        "country": ["country_name", "adm0_name"],
        "obs_type": ["spatialprecision_km_", "observation_type",
                     "data_type", "spatial_type"],
    }

    for canonical, options in aliases.items():
        base = df.get(canonical, pd.Series(pd.NA, index=df.index))
        for alt in options:
            if alt in df.columns:
                base = base.combine_first(df[alt])
        df[canonical] = base

    if "country" not in df.columns and "_source_file" in df.columns:
        def infer(src):
            for c in CANDIDATE_COUNTRIES + ["Ethiopia", "Uganda", "Ghana", "Zambia", "Mali"]:
                if c.lower() in src.lower():
                    return c
            return "Unknown"
        df["country"] = df["_source_file"].apply(infer)

    return df


def filter_dataset(df):
    """Apply study-scope filters and report counts."""
    n0 = len(df)
    print(f"Filter steps (starting n={n0:,}):")

    if "year" in df.columns:
        df = df[df["year"].between(YEAR_MIN, YEAR_MAX)]
        print(f"  After year filter ({YEAR_MIN}-{YEAR_MAX}): {len(df):,}")

    if "crop_name" in df.columns:
        df = df[df["crop_name"].str.lower().str.contains(PRIMARY_CROP, na=False)]
        print(f"  After crop filter (maize): {len(df):,}")

    if "obs_type" in df.columns:
        obs_dtype = pd.api.types.infer_dtype(df["obs_type"], skipna=True)
        if obs_dtype in ["integer", "floating"]:
            df = df[df["obs_type"].isna() | (df["obs_type"] <= 5)]
            print(f"  After spatial-precision filter (<=5 km): {len(df):,}")
        elif obs_dtype == "string" or obs_dtype == "object":
            mask = df["obs_type"].fillna("").astype(str).str.lower().str.contains("point|plot|field|gps", na=False)
            df = df[mask]
            print(f"  After point-level filter: {len(df):,}")
        else:
            print(f"  WARNING: obs_type has dtype {obs_dtype}, skipping filter")

    if "lat" in df.columns and "lon" in df.columns:
        df = df.dropna(subset=["lat", "lon"])
        df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
        print(f"  After GPS validity: {len(df):,}")

    if "yield_kgha" in df.columns:
        if df["yield_kgha"].median() < 20:
            print("  NOTE: yield appears to be in t/ha, converting to kg/ha...")
            df["yield_kgha"] = df["yield_kgha"] * 1000
        df = df[df["yield_kgha"] > 0]
        print(f"  After positive yield: {len(df):,}")

    if "country" in df.columns:
        df["country"] = df["country"].str.strip().str.title()
        df = df[df["country"].isin(CANDIDATE_COUNTRIES)]
        print(f"  After country filter {CANDIDATE_COUNTRIES}: {len(df):,}\n")

    return df


def check_viable_countries(df):
    """Report per-country counts."""
    if "country" not in df.columns:
        return []

    counts = df.groupby("country").size().sort_values(ascending=False)
    print("Per-country counts (before cloud/cropland filters):")
    for country, n in counts.items():
        flag = "" if n >= MIN_OBSERVATIONS_PER_COUNTRY else " WARNING"
        print(f"  {country}: {n:6,}{flag}")

    viable = counts[counts >= MIN_OBSERVATIONS_PER_COUNTRY].index.tolist()
    if len(viable) < len(CANDIDATE_COUNTRIES):
        dropped = [c for c in CANDIDATE_COUNTRIES if c not in viable]
        print(f"  Dropped (threshold {MIN_OBSERVATIONS_PER_COUNTRY}): {dropped}")

    print(f"  Viable for LOCO CV: {viable}\n")
    return viable


def stratified_sample_country(df_country, target_n, seed):
    """Sample target_n rows using yield quantile x year strata."""
    rng = np.random.default_rng(seed)
    df = df_country.copy().reset_index(drop=True)

    if len(df) <= target_n:
        return df

    df["_yield_bin"] = pd.qcut(
        df["yield_kgha"], q=N_YIELD_BINS, labels=False, duplicates="drop"
    )
    years = sorted(df["year"].dropna().unique())
    strata_keys = [(yb, yr) for yb in df["_yield_bin"].dropna().unique() for yr in years]

    n_strata = len(strata_keys)
    base_quota = target_n // n_strata
    remainder = target_n - base_quota * n_strata

    strata_sizes = {
        k: len(df[(df["_yield_bin"] == k[0]) & (df["year"] == k[1])])
        for k in strata_keys
    }
    sorted_keys = sorted(strata_keys, key=lambda k: strata_sizes[k], reverse=True)

    selected_indices = []
    for i, (yb, yr) in enumerate(sorted_keys):
        quota = base_quota + (1 if i < remainder else 0)
        stratum_idx = df[(df["_yield_bin"] == yb) & (df["year"] == yr)].index.tolist()
        if stratum_idx:
            n_take = min(quota, len(stratum_idx))
            chosen = rng.choice(stratum_idx, size=n_take, replace=False)
            selected_indices.extend(chosen.tolist())

    if len(selected_indices) < target_n:
        remaining = list(set(df.index) - set(selected_indices))
        shortfall = target_n - len(selected_indices)
        extra = rng.choice(remaining, size=min(shortfall, len(remaining)), replace=False)
        selected_indices.extend(extra.tolist())

    result = df.loc[sorted(selected_indices)].drop(columns=["_yield_bin"])
    return result


def main():
    df = load_grow_africa_from_zip(ZENODO_ZIP)
    df = normalise_columns(df)
    df = filter_dataset(df)
    viable = check_viable_countries(df)

    df = df.reset_index(drop=True)
    df["field_id"] = (
        df["country"].str[:3].str.lower()
        + "_"
        + df.index.astype(str).str.zfill(6)
    )

    countries = sorted(df["country"].unique())
    n_countries = len(countries)
    target_per_country = TARGET_TOTAL // n_countries
    remainder_countries = TARGET_TOTAL - target_per_country * n_countries

    print(f"Stratified sampling: {TARGET_TOTAL:,} total across {n_countries} countries")
    print(f"  Base: {target_per_country}/country, remainder: {remainder_countries}\n")

    country_sizes = df.groupby("country").size().sort_values()
    frames = []
    report_rows = []

    for i, country in enumerate(country_sizes.index):
        target_n = target_per_country + (1 if i < remainder_countries else 0)
        country_df = df[df["country"] == country]
        sampled = stratified_sample_country(country_df, target_n, seed=RANDOM_SEED + i)
        frames.append(sampled)

        for yr in sorted(country_df["year"].dropna().unique()):
            n_before = len(country_df[country_df["year"] == yr])
            n_after = len(sampled[sampled["year"] == yr])
            report_rows.append({
                "country": country, "year": int(yr),
                "n_original": n_before, "n_sampled": n_after,
            })

        pct = len(sampled) / len(country_df) * 100
        print(f"  {country:12s}: {len(country_df):6,} -> {len(sampled):5,} ({pct:5.1f}%)")

    sampled_df = pd.concat(frames, ignore_index=True)

    assert sampled_df["field_id"].nunique() == len(sampled_df), "Duplicate field_ids"
    assert sampled_df["yield_kgha"].isna().sum() == 0, "NaN yield"
    assert not sampled_df[["lat", "lon"]].isna().any().any(), "NaN GPS"

    sampled_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nOK: {OUTPUT_PATH} ({len(sampled_df):,} rows)\n")

    counts_df = sampled_df.groupby("country").agg(
        n=("field_id", "count"),
        yield_mean=("yield_kgha", "mean"),
        yield_sd=("yield_kgha", "std"),
        yield_min=("yield_kgha", "min"),
        yield_max=("yield_kgha", "max"),
    ).round(1)
    counts_df.to_csv(COUNTS_PATH)
    print(f"OK: {COUNTS_PATH}")
    print(counts_df)
    print()

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(REPORT_PATH, index=False)
    print(f"OK: {REPORT_PATH}\n")

    print("Phase 1 complete: labels sampled and filtered.")
    return sampled_df


if __name__ == "__main__":
    main()
