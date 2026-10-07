"""
backend/inference.py
--------------------
Serves day-ahead forecasts and schedules for the days of the held-out test
month, using the trained ensemble saved by scripts/run_experiments.py.

A "day" is a forecast issued at 00:00 for hours 00-23 of that day, using only
data observed before midnight.
"""

import json
import threading

import numpy as np
import pandas as pd

import config
from models.forecaster import Forecaster, train_od_trips
from optimize.frequency import FrequencyOptimizer
from utils.dataset import prepare_data
from utils.route import ODLoadModel


class TransitInferenceEngine:
    def __init__(self, split: str = "test"):
        if not config.DATASET_PATH.exists():
            raise FileNotFoundError(f"{config.DATASET_PATH} not found — see README 'Data'.")
        meta_path = config.ARTIFACT_DIR / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No trained models in {config.ARTIFACT_DIR}. Run: python scripts/run_experiments.py")
        meta = json.loads(meta_path.read_text())
        df = pd.read_parquet(config.DATASET_PATH)
        self.data = prepare_data(df, meta["val_start"], meta["test_start"],
                                 meta["lookback"], meta["horizon"], stops=meta["stops"], verbose=False)
        self.forecaster = Forecaster(self.data)

        origins = self.data.splits[split]
        self.day_origins = origins[self.data.times[origins].hour == 0]
        self.days = [str(self.data.times[o].date()) for o in self.day_origins]

        # Load model covers every stop on the route, including those too quiet
        # to forecast; their boardings come from the training-period profile.
        trips = train_od_trips(self.data)
        self.route_stops = config.ROUTE_STOP_ORDER
        self.load_model = ODLoadModel(self.route_stops, config.ROUTE_IS_LOOP).fit(trips)
        hourly = df.assign(slot=pd.to_datetime(df["slot"]).dt.floor("h"))
        hourly = hourly[hourly["slot"] < pd.Timestamp(meta["val_start"])]
        hourly = hourly.groupby(["stop_id", "slot"])["ridership"].sum().reset_index()
        self.quiet_profile = (hourly.assign(how=hourly["slot"].dt.dayofweek * 24 + hourly["slot"].dt.hour)
                                    .groupby(["stop_id", "how"])["ridership"].mean())
        self._cache = {}
        self._lock = threading.Lock()

    def day_forecast(self, day_index: int):
        """(actual, forecast) boardings [N_model_stops, 24] and target times."""
        if not 0 <= day_index < len(self.day_origins):
            raise IndexError(f"day_index must be in [0, {len(self.day_origins) - 1}]")
        with self._lock:
            if day_index not in self._cache:
                o = self.day_origins[day_index:day_index + 1]
                self._cache[day_index] = (self.data.window(o, "y_raw")[0], self.forecaster.predict(o)[0])
        o = self.day_origins[day_index]
        return (*self._cache[day_index], self.data.times[o:o + self.data.horizon])

    def route_boardings(self, model_boardings: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
        """Boardings for every route stop in route order [n_route_stops, 24]."""
        how = times.dayofweek * 24 + times.hour
        out = np.zeros((len(self.route_stops), len(times)))
        pos = {s: i for i, s in enumerate(self.data.stops)}
        for r, stop in enumerate(self.route_stops):
            if stop in pos:
                out[r] = model_boardings[pos[stop]]
            else:
                out[r] = [self.quiet_profile.get((stop, h), 0.0) for h in how]
        return out

    def run_simulation(self, day_index: int, bus_capacity: int, fixed_headway: int) -> dict:
        actual, predicted, times = self.day_forecast(day_index)
        hours = list(times.hour)
        board = self.route_boardings(predicted, times)
        load = self.load_model.segment_load(board, hours)
        alight = self.load_model.alighting(board, hours)

        opt = FrequencyOptimizer(bus_capacity=bus_capacity, load_factor=config.PLANNING_LOAD_FACTOR,
                                 min_headway=config.MIN_HEADWAY_MIN,
                                 max_headway=config.MAX_HEADWAY_MIN, round_trip_min=config.ROUND_TRIP_MIN,
                                 cost_per_vehicle_hour=config.COST_PER_VEH_HOUR)
        actual_board = self.route_boardings(actual, times)
        actual_load = self.load_model.segment_load(actual_board, hours)
        res = opt.optimize(load, board.sum(axis=0), fixed_headway=fixed_headway,
                           actual_load=actual_load, actual_boardings=actual_board.sum(axis=0))

        err = np.abs(predicted - actual).sum() / max(actual.sum(), 1e-8) * 100

        def plan(p):
            return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}

        return {
            "day": self.days[day_index],
            "day_index": day_index,
            "n_days": len(self.days),
            "forecast": {
                "hours": hours,
                "actual": actual.sum(axis=0).round(1).tolist(),
                "predicted": predicted.sum(axis=0).round(1).tolist(),
                "day_wmape": round(float(err), 2),
            },
            "optimization": {
                "assumptions": {"bus_capacity": bus_capacity, "fixed_headway": fixed_headway,
                                "round_trip_min": config.ROUND_TRIP_MIN,
                                "cost_per_vehicle_hour": config.COST_PER_VEH_HOUR,
                                "min_headway": config.MIN_HEADWAY_MIN, "max_headway": config.MAX_HEADWAY_MIN,
                                "planning_load_factor": config.PLANNING_LOAD_FACTOR},
                "fixed": plan(res["fixed"]),
                "optimized": plan(res["optimized"]),
                "savings": res["savings"],
                "peak_load": res["peak_load"].round(1).tolist(),
                "peak_segment": [self.load_model.segments[i] for i in res["peak_segment"]],
                "actual_peak_load": actual_load.max(axis=0).round(1).tolist(),
                "actual_peak_segment": [self.load_model.segments[i] for i in actual_load.argmax(axis=0)],
            },
            "heatmap": {"segments": self.load_model.segments, "hours": hours,
                        "data": load.round(1).tolist(), "actual": actual_load.round(1).tolist()},
            "stops": [{"stop": s, "boarding": round(float(b), 1), "alighting": round(float(a), 1)}
                      for s, b, a in zip(self.route_stops, board.sum(axis=1), alight.sum(axis=1))],
        }


_engines = {}
_engine_lock = threading.Lock()


def get_engine(split: str = "test") -> TransitInferenceEngine:
    with _engine_lock:
        if split not in _engines:
            _engines[split] = TransitInferenceEngine(split)
    return _engines[split]
