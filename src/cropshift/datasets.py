"""Analysis table: field labels, spectral and rainfall features, frozen embeddings.

Two analysis sets:
    precise   fields whose GROW-Africa spatial precision is 0.03 km (30 m); four countries
    all       every field that passed the extraction filters; five countries
Fields that share coordinates share a location id; every split groups by location id.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TARGET = "yield_kgha"
PRECISE_KM = 0.03
SPECTRAL = ["S2_B2", "S2_B3", "S2_B4", "S2_B5", "S2_B6", "S2_B7", "S2_B8", "S2_B8A", "S2_B11", "S2_B12",
            "NDVI", "EVI", "LSWI", "NDWI", "chirps_total", "chirps_mean", "chirps_cv"]
EMBEDDINGS = ("prithvi", "vit")
FEATURE_SETS = ["spectral", "prithvi", "vit", "spectral+prithvi"]


def location_id(df: pd.DataFrame, decimals: int = 5) -> np.ndarray:
    key = df["lat"].round(decimals).astype(str) + "_" + df["lon"].round(decimals).astype(str)
    return pd.factorize(key)[0]


def spatial_block(df: pd.DataFrame, degrees: float = 0.1) -> np.ndarray:
    """Grid cell id (about 11 km at the equator for 0.1 degrees)."""
    key = np.floor(df["lat"] / degrees).astype(int).astype(str) + "_" + np.floor(df["lon"] / degrees).astype(int).astype(str)
    return pd.factorize(key)[0]


def add_embeddings(df: pd.DataFrame, emb: pd.DataFrame, name: str) -> pd.DataFrame:
    """Attach one embedding table; every field must have exactly one embedding row."""
    missing = int((~df["field_id"].isin(emb["field_id"])).sum())
    if missing:
        raise ValueError(f"{name} embeddings are missing for {missing} of {len(df)} fields; "
                         f"finish scripts/04_extract_embeddings.py --model {name} first")
    return df.merge(emb, on="field_id", how="left", validate="one_to_one")


def load(subset: str = "precise", embeddings: tuple[str, ...] = EMBEDDINGS) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "master_features.parquet")
    for name in embeddings:
        df = add_embeddings(df, pd.read_parquet(PROCESSED / f"embeddings_{name}.parquet"), name)
    if subset == "precise":
        df = df[df["precision_km"] <= PRECISE_KM]
    elif subset != "all":
        raise ValueError(subset)
    df = df.sort_values(["country", "field_id"]).reset_index(drop=True)
    return df.assign(location=location_id(df), block=spatial_block(df))


def features(df: pd.DataFrame, name: str) -> list[str]:
    columns = []
    for part in name.split("+"):
        columns += SPECTRAL if part == "spectral" else [c for c in df.columns if c.startswith(f"{part}_")]
    return columns
