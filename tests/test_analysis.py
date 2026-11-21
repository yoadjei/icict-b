import numpy as np
import pandas as pd
import pytest

from cropshift import calibration, datasets, evaluation, shift


def test_scores_and_murphy_decomposition_add_up():
    rng = np.random.default_rng(0)
    y = rng.normal(3000, 1500, 500)
    p = 0.4 * y + rng.normal(800, 600, 500)
    s = evaluation.scores(y, p)
    m = evaluation.murphy(y, p)
    assert np.isclose(m["mse"], s["rmse"] ** 2)
    assert np.isclose(m["offset"], s["bias"] ** 2)
    assert np.isclose(m["offset_share"] + m["scale_share"] + m["unexplained_share"], 1.0)
    assert np.isclose(s["r2"], 1 - m["mse"] / np.var(y))


def test_r2_is_zero_for_the_evaluated_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(evaluation.scores(y, np.full(4, y.mean()))["r2"], 0.0)


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 300)
    p = y + rng.normal(0, 0.5, 300)
    groups = np.repeat(np.arange(100), 3)
    lo, hi = evaluation.bootstrap_interval(y, p, groups, "r2", n_boot=200)
    assert lo < evaluation.scores(y, p)["r2"] < hi


def test_calibration_draw_never_shares_a_location():
    rng = np.random.default_rng(2)
    locations = np.repeat(np.arange(40), 3)
    for k in (1, 10, 25, 50):
        cal, ev = calibration.draw(locations, k, rng)
        assert len(cal) == k
        assert not set(locations[cal]) & set(locations[ev])
        assert len(np.intersect1d(cal, ev)) == 0


def test_offset_and_linear_strategies_recover_a_known_shift():
    rng = np.random.default_rng(3)
    truth = rng.normal(2000, 500, 300)
    base = 0.5 * truth + 100  # transferred model: wrong scale and level
    X = np.zeros((300, 1))
    args = dict(model="ridge", X_source=X, y_source=truth, X_cal=X[:50], X_eval=X[50:])
    linear = calibration.predict("linear", base_cal=base[:50], base_eval=base[50:], y_cal=truth[:50], **args)
    assert np.allclose(linear, truth[50:])
    offset = calibration.predict("offset", base_cal=base[:50], base_eval=base[50:], y_cal=truth[:50], **args)
    assert abs(np.mean(offset - truth[50:])) < 0.1 * abs(np.mean(base[50:] - truth[50:]))
    local_mean = calibration.predict("local_mean", base_cal=base[:50], base_eval=base[50:], y_cal=truth[:50], **args)
    assert np.allclose(local_mean, truth[:50].mean())


def test_domain_auc_detects_shift_and_its_absence():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, (300, 3)), rng.normal(0, 1, (300, 3))
    groups = np.arange(300)
    assert abs(shift.domain_auc(a, b, groups, groups) - 0.5) < 0.1
    assert shift.domain_auc(a, b + 3.0, groups, groups) > 0.95


def test_location_ids_and_feature_selection():
    df = pd.DataFrame({"lat": [1.0, 1.0, 1.00002, 2.0], "lon": [3.0, 3.0, 3.0, 4.0],
                       **{c: 0.0 for c in datasets.SPECTRAL}, "prithvi_0": 0.0, "prithvi_1": 0.0, "vit_0": 0.0})
    ids = datasets.location_id(df)
    assert ids[0] == ids[1] and ids[0] != ids[2] and ids[2] != ids[3]
    assert datasets.features(df, "spectral") == datasets.SPECTRAL
    assert datasets.features(df, "spectral+prithvi") == datasets.SPECTRAL + ["prithvi_0", "prithvi_1"]
    assert datasets.features(df, "vit") == ["vit_0"]


@pytest.mark.parametrize("strategy", calibration.STRATEGIES)
def test_every_strategy_returns_one_prediction_per_evaluation_field(strategy):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(120, 4))
    y = X @ np.array([100.0, -50.0, 20.0, 0.0]) + 2000
    out = calibration.predict(strategy, "ridge", base_cal=y[:20] + 300, base_eval=y[20:] + 300, y_cal=y[:20],
                              X_source=X, y_source=y, X_cal=X[:20], X_eval=X[20:])
    assert out.shape == (100,) and np.all(np.isfinite(out))


def test_missing_embeddings_raise_instead_of_dropping_fields():
    df = pd.DataFrame({"field_id": ["a", "b", "c"], "yield_kgha": [1.0, 2.0, 3.0]})
    full = pd.DataFrame({"field_id": ["c", "a", "b"], "vit_0": [0.3, 0.1, 0.2]})
    merged = datasets.add_embeddings(df, full, "vit")
    assert list(merged["vit_0"]) == [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="missing for 1 of 3"):
        datasets.add_embeddings(df, full.iloc[:2], "vit")


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_cached_frame_computes_once_and_reloads_the_same_values(tmp_path, suffix):
    from cropshift import cache
    calls = []

    def compute():
        calls.append(1)
        return pd.DataFrame({"field_id": ["a", "b"], "pred": [1234.5678912345, -0.000123456789]})

    path = tmp_path / "nested" / f"step{suffix}"
    first = cache.cached_frame(path, compute)
    second = cache.cached_frame(path, compute)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)
    assert not list(path.parent.glob("*.tmp"))
