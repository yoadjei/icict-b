import datetime as dt

import numpy as np
import pandas as pd

from cropshift import earth_engine as ge
from cropshift import embeddings, features


def _excel(d: dt.date) -> float:
    return float((d - dt.date(1899, 12, 30)).days)


def test_harvest_year_follows_the_season_calendar():
    nigeria = pd.Series({"country": "Nigeria", "harvestyear": np.nan, "agyearstart": 2018, "agyearend": 2019, "year": 2019})
    malawi = pd.Series({"country": "Malawi", "harvestyear": np.nan, "agyearstart": 2018, "agyearend": 2019, "year": 2019})
    kenya = pd.Series({"country": "Kenya", "harvestyear": 2020.0, "year": 2020})
    assert ge.harvest_year(nigeria) == 2018  # April to September season, harvested in the start year
    assert ge.harvest_year(malawi) == 2019   # November to April season, harvested in the end year
    assert ge.harvest_year(kenya) == 2020


def test_season_window_prefers_recorded_dates():
    row = pd.Series({"country": "Kenya", "harvestyear": 2019.0, "year": 2019,
                     "plantdate": _excel(dt.date(2019, 3, 15)), "harvestdate": _excel(dt.date(2019, 8, 20))})
    w = ge.season_window(row)
    assert w.source == "plant_to_harvest" and w.start == dt.date(2019, 3, 15) and w.end == dt.date(2019, 8, 21)
    no_plant = row.drop("plantdate")
    assert ge.season_window(no_plant).source == "harvest_minus_150d"
    calendar = pd.Series({"country": "Malawi", "harvestyear": 2020.0, "year": 2020})
    w = ge.season_window(calendar)
    assert w.source == "calendar" and w.start == dt.date(2019, 11, 1) and w.end == dt.date(2020, 5, 1)


def test_spectral_indices_match_their_formulas():
    m = {f"S2_{b}": v for b, v in zip(ge.S2_BANDS, [0.05, 0.08, 0.06, 0.1, 0.2, 0.25, 0.3, 0.32, 0.2, 0.12])}
    out = features.spectral_indices(m)
    assert np.isclose(out["NDVI"], (0.3 - 0.06) / (0.3 + 0.06))
    assert np.isclose(out["EVI"], 2.5 * (0.3 - 0.06) / (0.3 + 6 * 0.06 - 7.5 * 0.05 + 1))


def test_band_medians_ignore_masked_pixels():
    bands = np.ones((4, 4, len(ge.S2_BANDS)))
    bands[0, 0] = 100.0
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    assert all(v == 1.0 for v in features.band_medians(bands, valid).values())


def test_embedding_inputs_have_the_model_shapes():
    patch = embeddings.fill_masked(np.full((64, 64, len(ge.S2_BANDS)), 0.1, dtype=np.float32),
                                   np.ones((64, 64), dtype=bool))
    assert embeddings.prithvi_input(patch).shape == (6, 224, 224)
    assert embeddings.vit_input(patch).shape == (3, 224, 224)
