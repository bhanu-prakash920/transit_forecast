"""
utils/metrics.py
----------------
Forecast error metrics on the original ridership scale (passengers/hour).

MAPE is undefined at zero and explodes near it, so it is reported only over
stop-hours whose actual boardings are at least MAPE_MIN_ACTUAL, and labelled
that way. WMAPE (= sum|err| / sum|actual|) is the headline metric.
"""

import numpy as np

MAPE_MIN_ACTUAL = 10.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0)
    err = y_pred - y_true
    mask = y_true >= MAPE_MIN_ACTUAL
    return {
        "MAE":   float(np.mean(np.abs(err))),
        "RMSE":  float(np.sqrt(np.mean(err ** 2))),
        "WMAPE": float(np.sum(np.abs(err)) / max(np.sum(np.abs(y_true)), 1e-8) * 100),
        f"MAPE(y>={MAPE_MIN_ACTUAL:g})": float(np.mean(np.abs(err[mask]) / y_true[mask]) * 100)
        if mask.any() else float("nan"),
    }


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return compute_metrics(y_true, y_pred)["WMAPE"]
