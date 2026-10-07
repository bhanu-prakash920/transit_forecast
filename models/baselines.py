"""
models/baselines.py
-------------------
Reference forecasters every deep model has to beat.

  seasonal_naive(24)   same hour yesterday
  seasonal_naive(168)  same hour last week
  hour_of_week_mean    training-period average for that stop and hour of week
  XGBoostDirect        one gradient-boosted model over (stop, origin, horizon)
                       rows using only information available at the origin

All functions return passenger forecasts shaped [S, N, H] for the given origins.
"""

import numpy as np
import xgboost as xgb

from utils.dataset import PreparedData, hour_of_week


def seasonal_naive(data: PreparedData, origins: np.ndarray, period: int) -> np.ndarray:
    idx = origins[:, None] + np.arange(data.horizon) - period
    return data.y_raw[:, idx].transpose(1, 0, 2)


def hour_of_week_mean(data: PreparedData, origins: np.ndarray) -> np.ndarray:
    train_end = data.times[data.splits["train"][-1] + data.horizon - 1]
    train_mask = data.times <= train_end
    how = hour_of_week(data.times)
    prof = np.stack([data.y_raw[:, train_mask & (how == h)].mean(axis=1) for h in range(168)], axis=1)
    idx = origins[:, None] + np.arange(data.horizon)
    return prof[:, how[idx]].transpose(1, 0, 2)


class XGBoostDirect:
    """Global direct multi-horizon model on tabular features (log1p target)."""

    PARAMS = dict(n_estimators=2000, max_depth=8, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                  tree_method="hist", early_stopping_rounds=50)

    def __init__(self, data: PreparedData, seed: int = 0):
        self.data = data
        self.model = xgb.XGBRegressor(**self.PARAMS, random_state=seed, n_jobs=-1)

    def features(self, origins: np.ndarray) -> np.ndarray:
        d = self.data
        S, N, H = len(origins), d.n_stops, d.horizon
        o = origins[:, None, None]
        s = np.arange(N)[None, :, None]
        h = np.arange(H)[None, None, :]
        t = o + h
        y = d.y_raw
        last = y[:, origins - 1].T[:, :, None]                          # [S, N, 1]
        last24 = np.stack([y[:, oo - 24:oo].mean(axis=1) for oo in origins])[:, :, None]
        last168 = np.stack([y[:, oo - 168:oo].mean(axis=1) for oo in origins])[:, :, None]
        fut = d.window(origins, "fut")                                  # [S, N, H, F]
        cols = [
            np.broadcast_to(s, (S, N, H)),
            np.broadcast_to(h, (S, N, H)),
            np.broadcast_to(d.times.hour.values[t], (S, N, H)),
            np.broadcast_to(d.times.dayofweek.values[t], (S, N, H)),
            y[s, t - 24], y[s, t - 168],
            np.broadcast_to(last, (S, N, H)),
            np.broadcast_to(last24, (S, N, H)),
            np.broadcast_to(last168, (S, N, H)),
        ]
        cols += [fut[..., k] for k in range(fut.shape[-1])]
        return np.stack(cols, axis=-1).reshape(S * N * H, -1).astype(np.float32)

    def _target(self, origins):
        return np.log1p(self.data.window(origins, "y_raw")).reshape(-1)

    def fit(self):
        tr, va = self.data.splits["train"], self.data.splits["val"]
        self.model.fit(self.features(tr), self._target(tr),
                       eval_set=[(self.features(va), self._target(va))], verbose=False)
        return self

    def predict(self, origins: np.ndarray) -> np.ndarray:
        p = np.expm1(self.model.predict(self.features(origins)))
        return np.maximum(p, 0).reshape(len(origins), self.data.n_stops, self.data.horizon)
