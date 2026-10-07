"""
scripts/evaluate_schedule.py
----------------------------
Scores the forecast-driven schedule on every day of a month against the demand
that actually happened, next to the fixed schedule and an oracle plan built
from actual demand (the best any forecast could do).

    python scripts/evaluate_schedule.py            # test month, config.PLANNING_LOAD_FACTOR
    python scripts/evaluate_schedule.py --tune     # choose the load factor on the validation month

--tune picks the cheapest planning load factor whose VALIDATION-month
overcrowding stays within 2% of service hours, then reports the test month.
Writes outputs/results/schedule_eval.csv and schedule_summary.json.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from backend.inference import get_engine  # noqa: E402
from optimize.frequency import FrequencyOptimizer  # noqa: E402

LOAD_FACTORS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
MAX_OVERCROWDED_SHARE = 0.02


def evaluate(split: str, load_factor: float, capacity=config.BUS_CAPACITY,
             fixed_headway=config.FIXED_HEADWAY_MIN) -> pd.DataFrame:
    engine = get_engine(split)
    kw = dict(bus_capacity=capacity, min_headway=config.MIN_HEADWAY_MIN, max_headway=config.MAX_HEADWAY_MIN,
              round_trip_min=config.ROUND_TRIP_MIN, cost_per_vehicle_hour=config.COST_PER_VEH_HOUR)
    planner = FrequencyOptimizer(load_factor=load_factor, **kw)
    oracle_planner = FrequencyOptimizer(load_factor=1.0, **kw)
    rows = []
    for i, day in enumerate(engine.days):
        actual, predicted, times = engine.day_forecast(i)
        hours = list(times.hour)
        a_board = engine.route_boardings(actual, times)
        p_board = engine.route_boardings(predicted, times)
        a_load = engine.load_model.segment_load(a_board, hours)
        p_load = engine.load_model.segment_load(p_board, hours)
        real = dict(fixed_headway=fixed_headway, actual_load=a_load, actual_boardings=a_board.sum(axis=0))
        planned = planner.optimize(p_load, p_board.sum(axis=0), **real)
        oracle = oracle_planner.optimize(a_load, a_board.sum(axis=0), **real)
        for name, plan in [("fixed", planned["fixed"]), ("forecast", planned["optimized"]),
                           ("oracle", oracle["optimized"])]:
            rows.append({"split": split, "load_factor": load_factor, "day": day, "plan": name,
                         "vehicle_hours": plan["vehicle_hours"], "cost": plan["cost"],
                         "overcrowded_hours": plan["realized_overcrowded_hours"],
                         "avg_wait_min": plan["realized_avg_wait_min"]})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby("plan").agg(vehicle_hours_per_day=("vehicle_hours", "mean"),
                               cost_per_day=("cost", "mean"),
                               overcrowded_hours_total=("overcrowded_hours", "sum"),
                               days_with_overcrowding=("overcrowded_hours", lambda x: int((x > 0).sum())),
                               avg_wait_min=("avg_wait_min", "mean")).reindex(["fixed", "forecast", "oracle"])
    s["cost_vs_fixed_pct"] = (s["cost_per_day"] / s.loc["fixed", "cost_per_day"] - 1) * 100
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tune", action="store_true")
    args = p.parse_args()

    load_factor, curve = config.PLANNING_LOAD_FACTOR, None
    if args.tune:
        rows = []
        for lf in LOAD_FACTORS:
            s = summarize(evaluate("val", lf)).loc["forecast"]
            rows.append({"load_factor": lf, **s.to_dict()})
        curve = pd.DataFrame(rows)
        n_hours = len(get_engine("val").days) * 24
        ok = curve[curve["overcrowded_hours_total"] <= MAX_OVERCROWDED_SHARE * n_hours]
        load_factor = float(ok.sort_values("cost_per_day").iloc[0]["load_factor"]) if len(ok) \
            else float(curve.sort_values("overcrowded_hours_total").iloc[0]["load_factor"])
        print("Validation month, forecast-driven plan per planning load factor:")
        print(curve.round(2).to_string(index=False))
        print(f"→ chosen load factor {load_factor} (set PLANNING_LOAD_FACTOR in config.py to use it in the API)\n")

    results = {}
    for lf in sorted({1.0, load_factor}, reverse=True):
        df = evaluate("test", lf)
        s = summarize(df)
        results[lf] = s
        print(f"Test month ({df['day'].nunique()} days), planning load factor {lf}, "
              f"scored against ACTUAL demand:")
        print(s.round(2).to_string(), "\n")
        if lf == load_factor:
            df.to_csv(config.RESULTS_DIR / "schedule_eval.csv", index=False)

    out = {"assumptions": {"bus_capacity": config.BUS_CAPACITY, "fixed_headway": config.FIXED_HEADWAY_MIN,
                           "round_trip_min": config.ROUND_TRIP_MIN,
                           "cost_per_vehicle_hour": config.COST_PER_VEH_HOUR},
           "chosen_load_factor": load_factor,
           "test": {str(lf): s.round(3).reset_index().to_dict("records") for lf, s in results.items()},
           "validation_curve": curve.round(3).to_dict("records") if curve is not None else None}
    (config.RESULTS_DIR / "schedule_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
