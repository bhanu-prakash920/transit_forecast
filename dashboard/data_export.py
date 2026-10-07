"""
dashboard/data_export.py
------------------------
Builds dashboard/data.json from the experiment results and trained models.
Every number the dashboard shows comes from here or from the live API; nothing
is hand-entered.

    python scripts/run_experiments.py
    python scripts/evaluate_schedule.py --tune
    python dashboard/data_export.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from backend.inference import get_engine  # noqa: E402

# Short description shown under each model name.
KIND = {
    "ensemble_resid": "Production model", "ensemble": "Validation-weighted blend",
    "xgb_direct": "Gradient boosting", "agcrn": "Learned graph",
    "stgnn_route": "Route graph", "stgnn_od": "Passenger-flow graph",
    "gru_nograph": "No graph", "stgnn_v3": "BiLSTM + attention",
    "how_mean": "Stop × hour-of-week average", "naive_168h": "Seasonal naive",
    "naive_24h": "Seasonal naive",
}


def error_breakdowns():
    z = np.load(config.RESULTS_DIR / "test_predictions.npz", allow_pickle=True)
    y, p = z["y_true"], z["y_pred"]
    err = np.abs(p - y)
    by_stop = err.sum(axis=(0, 2)) / np.maximum(y.sum(axis=(0, 2)), 1e-8) * 100
    hours = (pd.to_datetime(z["times"]).hour.values[:, None] + np.arange(y.shape[-1])) % 24
    e, a = err.sum(axis=1), y.sum(axis=1)
    by_hour = []
    for h in range(24):
        act = a[hours == h].sum()
        by_hour.append({"hour": h, "wmape": round(float(e[hours == h].sum() / act * 100), 2) if act > 0 else None})
    return ([{"stop": str(s), "wmape": round(float(w), 2)} for s, w in zip(z["stops"], by_stop)], by_hour)


def schedule_data():
    path = config.RESULTS_DIR / "schedule_summary.json"
    if not path.exists():
        raise FileNotFoundError("Run: python scripts/evaluate_schedule.py --tune")
    s = json.loads(path.read_text())
    lf = s["chosen_load_factor"]
    chosen = {r["plan"]: r for r in s["test"][str(lf)]}
    unbuffered = {r["plan"]: r for r in s["test"].get("1.0", [])}
    daily = pd.read_csv(config.RESULTS_DIR / "schedule_eval.csv")
    days = []
    for day, g in daily.groupby("day", sort=True):
        row = {"day": day}
        for _, r in g.iterrows():
            row[r["plan"]] = {"vehicle_hours": r["vehicle_hours"], "overcrowded_hours": int(r["overcrowded_hours"]),
                              "avg_wait_min": round(r["avg_wait_min"], 2)}
        days.append(row)
    return {"load_factor": lf, "assumptions": s["assumptions"], "plans": chosen,
            "unbuffered": unbuffered.get("forecast"), "validation_curve": s["validation_curve"], "days": days}


def main():
    summary = json.loads((config.RESULTS_DIR / "summary.json").read_text())
    metrics = pd.DataFrame(summary["metrics"])
    mape_col = [c for c in metrics.columns if c.startswith("MAPE(") and not c.endswith("_mean")][0]
    models = [{
        "id": r["id"], "name": r["model"], "kind": KIND.get(r["id"], ""),
        "wmape": round(r["WMAPE_mean"], 2), "wmape_text": r["WMAPE"],
        "mae": round(r["MAE_mean"], 2), "rmse": round(r["RMSE_mean"], 2),
        "mape": round(r[f"{mape_col}_mean"], 2), "seeds": int(r["seeds"]),
    } for r in metrics.sort_values("WMAPE_mean").to_dict("records")]

    by_stop, by_hour = error_breakdowns()
    engine = get_engine()
    dataset = pd.read_parquet(config.DATASET_PATH, columns=["slot"])
    trips = len(pd.read_parquet(config.OD_TRIPS_PATH, columns=["slot"])) if config.OD_TRIPS_PATH.exists() else 0

    data = {
        "meta": {
            "route": config.TARGET_ROUTE, "route_is_loop": config.ROUTE_IS_LOOP,
            "test_period": summary["test_period"], "seeds": summary["seeds"],
            "samples": summary["samples"],
            "split": {"val_start": config.VAL_START, "test_start": config.TEST_START},
            "n_forecast_stops": summary["n_stops"], "n_route_stops": len(config.ROUTE_STOP_ORDER),
            "od_trips": trips, "records": len(dataset),
            "months": int(pd.to_datetime(dataset["slot"]).dt.to_period("M").nunique()),
            "days": engine.days, "currency": "₹", "mape_label": mape_col,
            "defaults": {"bus_capacity": config.BUS_CAPACITY, "fixed_headway": config.FIXED_HEADWAY_MIN},
        },
        "models": models,
        "weights": {k: round(v, 3) for k, v in summary["weights"].items() if v > 0.005},
        "ablation": summary.get("ablation"),
        "errors": {"by_stop": by_stop, "by_hour": by_hour},
        "schedule": schedule_data(),
        "simulation": engine.run_simulation(0, config.BUS_CAPACITY, config.FIXED_HEADWAY_MIN),
    }
    out = config.DASHBOARD_DIR / "data.json"
    out.write_text(json.dumps(data, indent=1, default=float))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
