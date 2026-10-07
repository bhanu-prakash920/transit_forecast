"""
models/evaluate.py
------------------
Figures for the experiment report. All inputs are in passengers/hour; hours
on the x-axis are clock hours of the TARGET, not positions in the horizon.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name):
    config.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.PLOT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def _target_hours(origin_times: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    return (origin_times.hour.values[:, None] + np.arange(horizon)) % 24


def plot_model_comparison(metrics: pd.DataFrame):
    m = metrics.sort_values("WMAPE_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(m) + 1.2))
    colors = ["#2a6fdb" if i.startswith("ensemble") else "#9aa5b1" for i in m["id"]]
    ax.barh(m["model"], m["WMAPE_mean"], color=colors)
    for y, v in enumerate(m["WMAPE_mean"]):
        ax.text(v + 0.1, y, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("Test WMAPE (%) — lower is better")
    ax.set_title("Model comparison, test month")
    _save(fig, "model_comparison.png")


def plot_error_by_hour(y_true, y_pred, origin_times, name):
    hours = _target_hours(origin_times, y_true.shape[-1])            # [S, H]
    err = np.abs(y_pred - y_true).sum(axis=1)                         # [S, H] summed over stops
    act = y_true.sum(axis=1)
    by_hour = [err[hours == h].sum() / max(act[hours == h].sum(), 1e-8) * 100 for h in range(24)]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(24), by_hour, color="#2a6fdb")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("WMAPE (%)")
    ax.set_title(f"Error by hour of day — {name}")
    _save(fig, f"error_by_hour_{name}.png")


def plot_error_by_stop(y_true, y_pred, stops, name):
    err = np.abs(y_pred - y_true).sum(axis=(0, 2)) / np.maximum(y_true.sum(axis=(0, 2)), 1e-8) * 100
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(len(stops)), err, color="#2a6fdb")
    ax.set_xticks(range(len(stops)))
    ax.set_xticklabels(stops, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("WMAPE (%)")
    ax.set_title(f"Error by stop (route order) — {name}")
    _save(fig, f"error_by_stop_{name}.png")


def plot_forecast_day(y_true, y_pred, origin_times, stops, name, day_index=0):
    """Route-total and busiest-stop forecast for one midnight-origin test day."""
    mid = np.where(origin_times.hour == 0)[0]
    i = mid[min(day_index, len(mid) - 1)]
    busiest = int(y_true.sum(axis=(0, 2)).argmax())
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for ax, (t, p, title) in zip(axes, [
        (y_true[i].sum(0), y_pred[i].sum(0), "All stops"),
        (y_true[i, busiest], y_pred[i, busiest], stops[busiest]),
    ]):
        ax.plot(range(24), t, label="Actual", color="#1f2933", lw=2)
        ax.plot(range(24), p, label="Forecast", color="#2a6fdb", lw=2, ls="--")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Boardings / hour")
        ax.set_title(f"{title} — {origin_times[i].date()}")
        ax.legend()
    _save(fig, f"forecast_day_{name}.png")


def plot_training_history(history: dict, name: str):
    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax1.plot(history["train_loss"], color="#9aa5b1", label="train loss (WMAPE-scaled L1)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax2 = ax1.twinx()
    ax2.plot(history["val_wmape"], color="#2a6fdb", label="val WMAPE (%)")
    ax2.set_ylabel("Val WMAPE (%)")
    fig.legend(loc="upper right")
    ax1.set_title(f"Training history — {name}")
    _save(fig, f"history_{name}.png")
