"""
models/ensemble.py
------------------
Combines member forecasts (neural models, XGBoost, historical profile).

1. Convex weights minimising validation WMAPE (test data is never used).
2. Optional XGBoost residual correction. Residuals of the training period are
   in-sample for every member and much smaller than real errors, so the
   corrector is trained on VALIDATION residuals, which are out-of-sample:
     first 50% of val days → fit, next 25% → early stopping,
     last 25% → decide whether the correction is kept at all.
"""

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize

from utils.dataset import PreparedData
from utils.metrics import wmape


def fit_weights(preds: dict, y_true: np.ndarray) -> dict:
    names = list(preds)
    P = np.stack([preds[n] for n in names])                       # [M, S, N, H]

    def loss(w):
        return wmape(y_true, np.tensordot(w, P, axes=1))

    w0 = np.full(len(names), 1 / len(names))
    res = minimize(loss, w0, method="SLSQP", bounds=[(0, 1)] * len(names),
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    # SLSQP on a non-smooth objective can stall; never do worse than the best member.
    best_single = min(names, key=lambda n: wmape(y_true, preds[n]))
    if wmape(y_true, preds[best_single]) < loss(w):
        w = np.array([1.0 if n == best_single else 0.0 for n in names])
    return {n: float(x) for n, x in zip(names, w)}


def combine(preds: dict, weights: dict) -> np.ndarray:
    return sum(w * preds[n] for n, w in weights.items() if w > 0)


class ResidualCorrector:
    PARAMS = dict(n_estimators=1000, max_depth=4, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                  tree_method="hist", early_stopping_rounds=50)

    def __init__(self, data: PreparedData, member_names: list, seed: int = 0):
        self.data, self.members = data, list(member_names)
        self.model = xgb.XGBRegressor(**self.PARAMS, random_state=seed, n_jobs=-1)
        self.enabled = False
        self.report = {}

    def features(self, origins, preds: dict, ens: np.ndarray) -> np.ndarray:
        d = self.data
        S, N, H = ens.shape
        t = origins[:, None, None] + np.arange(H)[None, None, :]
        s = np.arange(N)[None, :, None]
        cols = [np.broadcast_to(s, (S, N, H)),
                np.broadcast_to(np.arange(H), (S, N, H)),
                np.broadcast_to(d.times.hour.values[t], (S, N, H)),
                np.broadcast_to(d.times.dayofweek.values[t], (S, N, H)),
                np.log1p(ens)]
        cols += [np.log1p(preds[m]) for m in self.members]
        cols += [np.log1p(d.y_raw[s, t - 24]), np.log1p(d.y_raw[s, t - 168])]
        return np.stack(cols, axis=-1).reshape(S * N * H, -1).astype(np.float32)

    def fit(self, origins, preds: dict, ens: np.ndarray, y_true: np.ndarray):
        days = np.unique(self.data.times[origins].normalize())
        cut1, cut2 = days[int(len(days) * 0.5)], days[int(len(days) * 0.75)]
        day = self.data.times[origins].normalize()
        parts = [day < cut1, (day >= cut1) & (day < cut2), day >= cut2]

        def sub(mask):
            return (self.features(origins[mask], {k: v[mask] for k, v in preds.items()}, ens[mask]),
                    (np.log1p(y_true[mask]) - np.log1p(ens[mask])).reshape(-1))

        (X_fit, r_fit), (X_es, r_es), (X_dec, _) = sub(parts[0]), sub(parts[1]), sub(parts[2])
        self.model.fit(X_fit, r_fit, eval_set=[(X_es, r_es)], verbose=False)

        dec = parts[2]
        before = wmape(y_true[dec], ens[dec])
        after = wmape(y_true[dec], self._apply(X_dec, ens[dec]))
        self.enabled = after < before - 0.1
        self.report = {"holdout_wmape_before": before, "holdout_wmape_after": after,
                       "enabled": bool(self.enabled)}
        return self

    def _apply(self, X, ens):
        corr = self.model.predict(X).reshape(ens.shape)
        return np.maximum(np.expm1(np.log1p(ens) + corr), 0)

    def predict(self, origins, preds: dict, ens: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return ens
        return self._apply(self.features(origins, preds, ens), ens)
