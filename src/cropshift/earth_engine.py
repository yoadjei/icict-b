"""Sentinel-2 patches and CHIRPS rainfall per field from Google Earth Engine.

For every labelled field we request two things in one worker call:
  * a 64 x 64 pixel (640 m) Sentinel-2 Level-1C median composite over the field's
    growing season window, with a per pixel validity band;
  * the CHIRPS daily precipitation series at the field over the same window.
Both are cached as one compressed .npz per field so the extraction is resumable.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Level-1C top of atmosphere reflectance is used for every field because Level-2A
# coverage over the study area is sparse before 2019 (for example 6 Level-2A versus
# 72 Level-1C scenes over the Nigerian sample location in 2018), which would remove
# Nigeria and most 2017 and 2018 seasons. Pixels are masked with Cloud Score+.
S2_COLLECTION = "COPERNICUS/S2_HARMONIZED"
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_BAND = "cs_cdf"
CLOUD_SCORE_MIN = 0.60  # keep pixels whose clear sky score is at least this value
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
PATCH_PX = 64
PIXEL_M = 10
DEG_PER_M = 1.0 / 111319.49079327357  # Earth Engine's equatorial degrees per metre
SCENE_CLOUD_MAX = 60  # scene level prefilter; pixel level masking does the real work
CHIRPS_SCALE_M = 5566
DAYS_BEFORE_HARVEST = 150
EXCEL_EPOCH = dt.date(1899, 12, 30)

# Calendar windows (start month, end month) used only when no field dates exist.
# The window always ends in the harvest year H; Malawi wraps the year boundary.
CALENDAR = {
    "Kenya": (3, 8),
    "Rwanda": (3, 8),
    "Tanzania": (3, 8),
    "Nigeria": (4, 9),
    "Malawi": (11, 4),
}


@dataclass(frozen=True)
class Window:
    start: dt.date
    end: dt.date  # exclusive
    source: str


def _excel_date(value: float) -> dt.date:
    return EXCEL_EPOCH + dt.timedelta(days=int(value))


def harvest_year(row: pd.Series) -> int:
    """Harvest year of the main season that produced the yield record.

    Field surveys and trials carry `harvestyear`. LSMS records carry an agricultural
    year (start, end): a season inside one calendar year is harvested in the start
    year (Nigeria 2018/19 wet season, Apr to Sep 2018); a season that wraps the year
    boundary is harvested in the end year (Malawi 2018/19, Nov 2018 to Apr 2019).
    """
    if pd.notna(row.get("harvestyear")):
        return int(row["harvestyear"])
    start_m, end_m = CALENDAR[row["country"]]
    wraps = start_m > end_m
    if pd.notna(row.get("agyearstart")) and pd.notna(row.get("agyearend")):
        return int(row["agyearend"] if wraps else row["agyearstart"])
    return int(row["year"])


def season_window(row: pd.Series) -> Window:
    """Growing season window for one field, preferring recorded field dates."""
    harvest = row.get("harvestdate")
    plant = row.get("plantdate")
    if pd.notna(harvest):
        end = _excel_date(harvest) + dt.timedelta(days=1)
        if pd.notna(plant) and 60 <= (end - _excel_date(plant)).days <= 270:
            return Window(_excel_date(plant), end, "plant_to_harvest")
        return Window(end - dt.timedelta(days=DAYS_BEFORE_HARVEST + 1), end, "harvest_minus_150d")
    year = harvest_year(row)
    start_m, end_m = CALENDAR[row["country"]]
    start_year = year - 1 if start_m > end_m else year
    start = dt.date(start_year, start_m, 1)
    end = dt.date(year + (end_m == 12), end_m % 12 + 1, 1)
    return Window(start, end, "calendar")


def _mask_and_scale(ee, image):
    clear = image.select(CLOUD_SCORE_BAND).gte(CLOUD_SCORE_MIN)
    scaled = image.select(S2_BANDS).updateMask(clear).divide(10000).toFloat()
    return scaled.copyProperties(image, ["system:time_start"])


def s2_composite(ee, lon: float, lat: float, window: Window):
    """Median composite with an all masked placeholder so empty seasons still have bands."""
    point = ee.Geometry.Point([lon, lat])
    placeholder = ee.Image.constant([0] * len(S2_BANDS)).rename(S2_BANDS).toFloat().updateMask(0)
    scenes = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(point)
        .filterDate(window.start.isoformat(), window.end.isoformat())
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", SCENE_CLOUD_MAX))
        .linkCollection(ee.ImageCollection(CLOUD_SCORE_COLLECTION), [CLOUD_SCORE_BAND])
        .map(lambda img: _mask_and_scale(ee, ee.Image(img)))
    )
    composite = scenes.merge(ee.ImageCollection([placeholder])).median()
    valid = composite.mask().reduce(ee.Reducer.min()).rename("valid")
    return composite.unmask(0).toFloat().addBands(valid.unmask(0).toFloat()), scenes.size()


def patch_grid(lon: float, lat: float) -> dict:
    step = PIXEL_M * DEG_PER_M
    half = PATCH_PX // 2
    return {
        "dimensions": {"width": PATCH_PX, "height": PATCH_PX},
        "affineTransform": {"scaleX": step, "shearX": 0, "translateX": lon - half * step,
                            "shearY": 0, "scaleY": -step, "translateY": lat + half * step},
        "crsCode": "EPSG:4326",
    }


def fetch_s2_patch(ee, lon: float, lat: float, window: Window) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (bands HxWx10 float32 reflectance, valid HxW uint8, n_scenes)."""
    image, n_scenes = s2_composite(ee, lon, lat, window)
    raw = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": patch_grid(lon, lat)})
    bands = np.stack([raw[b] for b in S2_BANDS], axis=-1).astype(np.float32)
    valid = (raw["valid"] > 0).astype(np.uint8)
    return bands, valid, int(n_scenes.getInfo())


def fetch_chirps(ee, lon: float, lat: float, window: Window) -> tuple[np.ndarray, np.ndarray]:
    """Daily CHIRPS precipitation (mm) at the field over the window: (dates as int64 ms, mm)."""
    series = (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterDate(window.start.isoformat(), window.end.isoformat())
        .select("precipitation")
        .getRegion(ee.Geometry.Point([lon, lat]), CHIRPS_SCALE_M)
        .getInfo()
    )
    header, rows = series[0], series[1:]
    t_idx, p_idx = header.index("time"), header.index("precipitation")
    rows = sorted((r[t_idx], r[p_idx]) for r in rows if r[p_idx] is not None)
    return np.array([r[0] for r in rows], dtype=np.int64), np.array([r[1] for r in rows], dtype=np.float32)
