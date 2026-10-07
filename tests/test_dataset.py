import numpy as np
import pandas as pd

from tests.conftest import SPLIT, make_df
from utils.dataset import WindowDataset, prepare_data


def test_lag_features_are_per_stop(df):
    d = prepare_data(df, stops=["A", "B", "C"], verbose=False, **SPLIT)
    lag = d.fut_cols.index("y_lag24")
    o = d.splits["test"][0]
    for s in range(d.n_stops):
        assert np.isclose(d.fut[s, o, lag], d.y_scaled[s, o - 24])
    # stops differ, so their lag features must differ too
    assert not np.allclose(d.fut[0, :, lag], d.fut[2, :, lag])


def test_splits_do_not_share_target_hours(df):
    d = prepare_data(df, stops=["A", "B", "C"], verbose=False, **SPLIT)
    hours = {k: set((v[:, None] + np.arange(d.horizon)).ravel()) for k, v in d.splits.items()}
    assert not hours["train"] & hours["val"]
    assert not hours["val"] & hours["test"]
    assert max(hours["train"]) < min(hours["val"]) <= max(hours["val"]) < min(hours["test"])


def test_statistics_ignore_future_data(df):
    """Changing validation/test ridership must not change scaler or profile."""
    d1 = prepare_data(df, stops=["A", "B", "C"], verbose=False, **SPLIT)
    df2 = df.copy()
    late = df2["slot"] >= pd.Timestamp(SPLIT["val_start"])
    df2.loc[late, "ridership"] *= 10
    d2 = prepare_data(df2, stops=["A", "B", "C"], verbose=False, **SPLIT)
    assert np.allclose(d1.scaler.log_max_, d2.scaler.log_max_)
    prof = d1.fut_cols.index("how_profile")
    assert np.allclose(d1.fut[..., prof], d2.fut[..., prof])


def test_stop_filter_uses_training_period():
    df = make_df(stops=("A", "B"))
    # B is silent in training and busy afterwards → must be dropped
    df.loc[(df["stop_id"] == "B") & (df["slot"] < SPLIT["val_start"]), "ridership"] = 0
    df.loc[(df["stop_id"] == "B") & (df["slot"] >= SPLIT["val_start"]), "ridership"] = 50
    d = prepare_data(df, verbose=False, **SPLIT)
    assert d.stops == ["A"]


def test_scaler_roundtrip(df):
    d = prepare_data(df, stops=["A", "B", "C"], verbose=False, **SPLIT)
    assert np.allclose(d.scaler.inverse(d.y_scaled), d.y_raw, atol=1e-3)


def test_window_shapes(df):
    d = prepare_data(df, stops=["A", "B", "C"], verbose=False, **SPLIT)
    ds = WindowDataset(d, d.splits["val"])
    x_hist, x_fut, y, o = ds[0]
    assert x_hist.shape == (3, d.lookback, len(d.hist_cols))
    assert x_fut.shape == (3, d.horizon, len(d.fut_cols))
    assert y.shape == (3, d.horizon)
    # history ends right before the first target hour
    assert np.isclose(x_hist[1, -1, 0], d.y_scaled[1, o - 1])
    assert np.isclose(y[1, 0], d.y_scaled[1, o])
