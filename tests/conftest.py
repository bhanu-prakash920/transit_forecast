import numpy as np
import pandas as pd
import pytest


def make_df(n_days=40, stops=("A", "B", "C"), seed=0):
    """Synthetic 15-min dataset with a daily cycle and different levels per stop."""
    rng = np.random.default_rng(seed)
    slots = pd.date_range("2020-01-01", periods=n_days * 96, freq="15min")
    rows = []
    for k, s in enumerate(stops):
        base = (k + 1) * 3 * (1 + np.clip(np.sin(2 * np.pi * (slots.hour.values - 6) / 24), 0, None))
        rows.append(pd.DataFrame({
            "slot": slots, "stop_id": s,
            "ridership": rng.poisson(base),
            "temperature": 20 + 5 * np.sin(2 * np.pi * slots.hour / 24),
            "precipitation": 0.0, "wind_speed": 10.0, "is_raining": 0,
            "is_holiday": 0, "is_day_off": (slots.dayofweek >= 5).astype(int),
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def df():
    return make_df()


SPLIT = dict(val_start="2020-01-29", test_start="2020-02-04")
