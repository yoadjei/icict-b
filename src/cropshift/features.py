"""Spectral and rainfall features from the cached per field Earth Engine extracts."""
from __future__ import annotations

import numpy as np

from cropshift.earth_engine import S2_BANDS

RAIN_BLOCK_DAYS = 30


def band_medians(bands: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Median reflectance of each band over the valid pixels of a patch."""
    mask = valid.astype(bool)
    if not mask.any():
        return {f"S2_{b}": np.nan for b in S2_BANDS}
    return {f"S2_{b}": float(np.median(bands[..., i][mask])) for i, b in enumerate(S2_BANDS)}


def _normalised_difference(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b) or (a + b) == 0:
        return np.nan
    return (a - b) / (a + b)


def spectral_indices(m: dict[str, float]) -> dict[str, float]:
    """NDVI, EVI, LSWI and NDWI from band medians, with the standard formulas."""
    b2, b3, b4, b8, b11 = (m[f"S2_{b}"] for b in ("B2", "B3", "B4", "B8", "B11"))
    evi_den = b8 + 6 * b4 - 7.5 * b2 + 1
    evi = 2.5 * (b8 - b4) / evi_den if np.isfinite(evi_den) and evi_den != 0 else np.nan
    return {
        "NDVI": _normalised_difference(b8, b4),
        "EVI": evi,
        "LSWI": _normalised_difference(b8, b11),
        "NDWI": _normalised_difference(b3, b8),
    }


def rainfall_stats(daily_mm: np.ndarray) -> dict[str, float]:
    """Seasonal total (mm), mean per 30 day block (mm) and CV across 30 day blocks."""
    if daily_mm.size == 0:
        return {"chirps_total": np.nan, "chirps_mean": np.nan, "chirps_cv": np.nan}
    n_blocks = max(1, daily_mm.size // RAIN_BLOCK_DAYS)
    blocks = np.array([chunk.sum() for chunk in np.array_split(daily_mm, n_blocks)])
    mean = float(blocks.mean())
    cv = float(blocks.std(ddof=0) / mean) if mean > 0 else np.nan
    return {"chirps_total": float(daily_mm.sum()), "chirps_mean": mean, "chirps_cv": cv}


def field_features(npz) -> dict[str, float]:
    """All tabular features for one cached field extract."""
    valid = npz["valid"]
    medians = band_medians(npz["bands"], valid)
    return {
        **medians,
        **spectral_indices(medians),
        **rainfall_stats(npz["chirps_mm"].astype(float)),
        "valid_fraction": float(valid.mean()),
        "n_scenes": int(npz["n_scenes"]),
        "window_start": str(npz["window_start"]),
        "window_end": str(npz["window_end"]),
        "window_source": str(npz["window_source"]),
    }


SPECTRAL_COLUMNS = [f"S2_{b}" for b in S2_BANDS] + ["NDVI", "EVI", "LSWI", "NDWI",
                                                     "chirps_total", "chirps_mean", "chirps_cv"]
