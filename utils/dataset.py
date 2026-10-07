"""
utils/dataset.py
----------------
Turns the processed 15-min dataset into leakage-free forecasting windows.

A sample is identified by its forecast origin o (an hourly index). Given data
up to hour o-1 the model forecasts boardings for hours o .. o+H-1 at every stop.

Inputs per sample
  x_hist [N, L, F_hist]  observed history (hours o-L .. o-1), per stop
  x_fut  [N, H, F_fut]   information known at forecast time about each target
                         hour: calendar, boardings 24 h and 168 h earlier, and
                         the stop's training-period hour-of-week profile
  y      [N, H]          scaled boardings to predict

Leakage rules
  - Samples are assigned to train/val/test by their TARGET hours: every target
    hour of a sample lies inside its split, so no target is shared across splits.
  - Stop filter, scalers and the hour-of-week profile use training hours only.
  - Weather is used for the history window only (future weather would be an
    oracle; a real deployment would need a weather forecast).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config

CAL_COLS = ["sin_hour", "cos_hour", "sin_dow", "cos_dow", "is_day_off", "is_holiday"]
WEATHER_COLS = ["temperature", "precipitation", "wind_speed", "is_raining"]


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """15-min rows → hourly rows (boardings summed, exogenous columns averaged)."""
    df = df.copy()
    df["slot"] = pd.to_datetime(df["slot"]).dt.floor("h")
    agg = {"ridership": "sum"}
    for c in WEATHER_COLS + ["is_holiday", "is_day_off", "is_makeup_workday"]:
        if c in df.columns:
            agg[c] = "mean" if c in WEATHER_COLS else "first"
    return df.groupby(["stop_id", "slot"], as_index=False).agg(agg)


def select_stops(hourly: pd.DataFrame, train_end: pd.Timestamp,
                 min_ridership: float = config.MIN_STOP_RIDERSHIP,
                 order: list = config.ROUTE_STOP_ORDER) -> list:
    """Stops with mean hourly boardings ≥ min_ridership in the TRAINING period, in route order."""
    means = hourly[hourly["slot"] < train_end].groupby("stop_id")["ridership"].mean()
    keep = set(means[means >= min_ridership].index)
    ordered = [s for s in order if s in keep]
    return ordered + sorted(keep - set(ordered))


class Log1pScaler:
    """
    Per-stop y_scaled = log1p(y) / log1p(max_train(y)).

    Fit on training hours only; later periods may slightly exceed 1.
    """

    def fit(self, y_train: np.ndarray) -> "Log1pScaler":       # [N, T_train]
        self.log_max_ = np.log1p(np.maximum(y_train.max(axis=1), 1.0)).astype(np.float32)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:           # [N, T]
        return (np.log1p(np.maximum(y, 0)) / self.log_max_[:, None]).astype(np.float32)

    def inverse(self, y_scaled: np.ndarray) -> np.ndarray:      # [..., N, T]
        return np.expm1(np.asarray(y_scaled) * self.log_max_[:, None])

    def state_dict(self) -> dict:
        return {"log_max": self.log_max_.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Log1pScaler":
        s = cls()
        s.log_max_ = np.asarray(d["log_max"], dtype=np.float32)
        return s


@dataclass
class PreparedData:
    stops: list
    times: pd.DatetimeIndex
    y_raw: np.ndarray             # [N, T]
    y_scaled: np.ndarray          # [N, T]
    hist: np.ndarray              # [N, T, F_hist]
    fut: np.ndarray               # [N, T, F_fut]  features describing hour t
    scaler: Log1pScaler
    hist_cols: list
    fut_cols: list
    splits: dict                  # name -> np.ndarray of origins
    lookback: int = config.LOOKBACK_HOURS
    horizon: int = config.FORECAST_HOURS
    weather_stats: dict = field(default_factory=dict)

    @property
    def n_stops(self): return len(self.stops)

    def window(self, origins: np.ndarray, part: str) -> np.ndarray:
        """Stacks one array over a set of origins → [S, N, len, ...]."""
        if part == "hist":
            idx = origins[:, None] + np.arange(-self.lookback, 0)
            return self.hist[:, idx].transpose(1, 0, 2, 3)
        idx = origins[:, None] + np.arange(self.horizon)
        src = {"fut": self.fut, "y": self.y_scaled, "y_raw": self.y_raw}[part]
        out = src[:, idx]
        return out.transpose(1, 0, 2, 3) if out.ndim == 4 else out.transpose(1, 0, 2)


def hour_of_week(times: pd.DatetimeIndex) -> np.ndarray:
    return times.dayofweek.values * 24 + times.hour.values


def prepare_data(df: pd.DataFrame,
                 val_start: str = config.VAL_START, test_start: str = config.TEST_START,
                 lookback: int = config.LOOKBACK_HOURS, horizon: int = config.FORECAST_HOURS,
                 stops: list = None, verbose: bool = True) -> PreparedData:
    val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)
    hourly = aggregate_to_hourly(df)
    stops = stops or select_stops(hourly, val_start)
    hourly = hourly[hourly["stop_id"].isin(stops)]

    times = pd.date_range(hourly["slot"].min(), hourly["slot"].max(), freq="h")
    y_raw = (hourly.pivot(index="stop_id", columns="slot", values="ridership")
                   .reindex(index=stops, columns=times).fillna(0).values.astype(np.float32))
    n, T = y_raw.shape
    train_mask = times < val_start

    scaler = Log1pScaler().fit(y_raw[:, train_mask])
    y_scaled = scaler.transform(y_raw)

    # Calendar (identical for all stops)
    exo = hourly.groupby("slot").first().reindex(times)
    cal = pd.DataFrame(index=times)
    cal["sin_hour"] = np.sin(2 * np.pi * times.hour / 24)
    cal["cos_hour"] = np.cos(2 * np.pi * times.hour / 24)
    cal["sin_dow"] = np.sin(2 * np.pi * times.dayofweek / 7)
    cal["cos_dow"] = np.cos(2 * np.pi * times.dayofweek / 7)
    cal["is_day_off"] = exo["is_day_off"].values if "is_day_off" in exo else (times.dayofweek >= 5)
    cal["is_holiday"] = exo["is_holiday"].values if "is_holiday" in exo else 0
    cal = cal.fillna(0).values.astype(np.float32)                         # [T, 6]

    # Weather, standardised with training statistics
    weather_cols = [c for c in WEATHER_COLS if c in exo]
    w_df = exo[weather_cols].astype(float).interpolate(limit_direction="both")
    if "precipitation" in w_df:
        w_df["precipitation"] = np.log1p(w_df["precipitation"].clip(lower=0))
    w = w_df.values
    mu, sd = w[train_mask].mean(axis=0), w[train_mask].std(axis=0) + 1e-6
    w = ((w - mu) / sd).astype(np.float32)

    # Lags (scaled). Values before the start of the series are unknown → 0 and
    # such origins are excluded below.
    def lag(a, k):
        out = np.zeros_like(a)
        out[:, k:] = a[:, :-k]
        return out
    lag24, lag168 = lag(y_scaled, 24), lag(y_scaled, 168)

    # Hour-of-week profile from training hours only. For a training hour the
    # hour itself is left out of its own average (leave-one-out), otherwise the
    # feature contains the target and the model learns to over-trust it.
    how = hour_of_week(times)
    prof_sum = np.zeros((n, 168))
    prof_cnt = np.zeros(168)
    for h in range(168):
        m = train_mask & (how == h)
        prof_sum[:, h] = y_raw[:, m].sum(axis=1)
        prof_cnt[h] = m.sum()
    prof_raw = prof_sum[:, how] / np.maximum(prof_cnt[how], 1)
    loo = (prof_sum[:, how] - y_raw) / np.maximum(prof_cnt[how] - 1, 1)
    prof_raw[:, train_mask] = loo[:, train_mask]
    profile = scaler.transform(prof_raw.astype(np.float32))              # [N, T]

    hist_cols = ["y"] + CAL_COLS + weather_cols + ["y_lag168"]
    hist = np.concatenate([
        y_scaled[..., None],
        np.broadcast_to(cal, (n, T, cal.shape[1])),
        np.broadcast_to(w, (n, T, w.shape[1])),
        lag168[..., None],
    ], axis=-1).astype(np.float32)

    fut_cols = CAL_COLS + ["y_lag24", "y_lag168", "how_profile"]
    fut = np.concatenate([
        np.broadcast_to(cal, (n, T, cal.shape[1])),
        lag24[..., None], lag168[..., None], profile[..., None],
    ], axis=-1).astype(np.float32)

    # Origins: need lookback history, a full week for lag168 of every input,
    # and all H target hours inside one split.
    first = max(lookback, 168 + lookback)
    origins = np.arange(first, T - horizon + 1)
    t_start, t_end = times[origins], times[origins + horizon - 1]
    splits = {
        "train": origins[t_end < val_start],
        "val":   origins[(t_start >= val_start) & (t_end < test_start)],
        "test":  origins[t_start >= test_start],
    }

    if verbose:
        dropped = sorted(set(df["stop_id"].unique()) - set(stops))
        print(f"Stops kept: {n} (dropped {dropped} — mean < {config.MIN_STOP_RIDERSHIP}/h in training period)")
        print(f"Hours: {T} ({times[0]} → {times[-1]})")
        print("Samples: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))

    return PreparedData(stops, times, y_raw, y_scaled, hist, fut, scaler,
                        hist_cols, fut_cols, splits, lookback, horizon,
                        {"cols": weather_cols, "mean": mu.tolist(), "std": sd.tolist()})


class WindowDataset(Dataset):
    """Yields (x_hist, x_fut, y, origin) for the given origins."""

    def __init__(self, data: PreparedData, origins: np.ndarray):
        self.origins = np.asarray(origins)
        self.x_hist = torch.from_numpy(np.ascontiguousarray(data.window(self.origins, "hist")))
        self.x_fut = torch.from_numpy(np.ascontiguousarray(data.window(self.origins, "fut")))
        self.y = torch.from_numpy(np.ascontiguousarray(data.window(self.origins, "y")))

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        return self.x_hist[i], self.x_fut[i], self.y[i], int(self.origins[i])


def get_dataloaders(data: PreparedData, batch_size: int = config.TRAIN["batch_size"],
                    seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        name: DataLoader(WindowDataset(data, data.splits[name]), batch_size=batch_size,
                         shuffle=(name == "train"), generator=g if name == "train" else None)
        for name in ("train", "val", "test")
    }
