"""
optimize/frequency.py
---------------------
Load-aware bus frequency planning for one route.

Input: forecast passenger flow over every route segment for each hour
(utils.route.ODLoadModel.segment_load) and forecast boardings.

Per hour:
  departures = max(ceil(peak segment load / (capacity × load_factor)), 60 / max_headway)
               capped at 60 / min_headway (the hour is then capacity-limited)
  headway    = 60 / departures
  vehicles   = ceil(round_trip_min / headway)     buses needed in service
Cost is vehicle-hours × cost_per_vehicle_hour. The round-trip time is not in
the data and is an explicit assumption (config.ROUND_TRIP_MIN).

Wait time: a passenger who fits on the next bus waits headway/2 on average;
one who does not fit waits one more headway. Averaged over boardings.
"""

import numpy as np


class FrequencyOptimizer:
    def __init__(self, bus_capacity: int = 80, load_factor: float = 1.0,
                 min_headway: float = 5, max_headway: float = 30,
                 round_trip_min: float = 60, cost_per_vehicle_hour: float = 50):
        if bus_capacity <= 0:
            raise ValueError("bus_capacity must be positive")
        if not 0 < load_factor <= 1.5:
            raise ValueError("load_factor must be in (0, 1.5]")
        if not 0 < min_headway <= max_headway <= 60:
            raise ValueError("need 0 < min_headway <= max_headway <= 60")
        if round_trip_min <= 0:
            raise ValueError("round_trip_min must be positive")
        self.capacity = bus_capacity
        self.max_load = bus_capacity * load_factor
        self.min_headway, self.max_headway = min_headway, max_headway
        self.round_trip = round_trip_min
        self.cost_rate = cost_per_vehicle_hour

    def _plan(self, peak, boardings, departures):
        departures = np.asarray(departures, dtype=float)
        headway = 60 / departures
        vehicles = np.ceil(self.round_trip / headway - 1e-9)
        hourly_capacity = departures * self.capacity
        overflow_share = np.where(peak > 0, np.clip(peak - hourly_capacity, 0, None) / np.maximum(peak, 1e-9), 0)
        wait = headway / 2 + overflow_share * headway
        total_board = boardings.sum()
        return {
            "departures": departures,
            "headway": headway,
            "vehicles": vehicles,
            "vehicle_hours": float(vehicles.sum()),
            "departures_total": float(departures.sum()),
            "cost": float(vehicles.sum() * self.cost_rate),
            "avg_wait_min": float((wait * boardings).sum() / total_board) if total_board > 0 else float(wait.mean()),
            "overcrowded_hours": int((peak > hourly_capacity + 1e-9).sum()),
            "max_utilisation": float((peak / hourly_capacity).max()),
        }

    def optimize(self, segment_load: np.ndarray, boardings: np.ndarray, fixed_headway: float = 15,
                 actual_load: np.ndarray = None, actual_boardings: np.ndarray = None) -> dict:
        """
        segment_load [n_segments, H]  forecast passengers crossing each segment per hour
        boardings    [H]              forecast total boardings per hour on the route
        actual_load, actual_boardings (optional): what really happened. Both plans
            are then also scored against it ("realized"), which is the honest
            test of a plan made from a forecast.
        """
        if fixed_headway <= 0 or fixed_headway > 60:
            raise ValueError("fixed_headway must be in (0, 60] minutes")
        segment_load = np.maximum(np.asarray(segment_load, dtype=float), 0)
        boardings = np.maximum(np.asarray(boardings, dtype=float), 0)
        peak = segment_load.max(axis=0)
        peak_segment = segment_load.argmax(axis=0)

        needed = np.ceil(peak / self.max_load - 1e-9)
        departures = np.clip(needed, 60 / self.max_headway, 60 / self.min_headway)
        optimized = self._plan(peak, boardings, departures)
        optimized["capacity_limited_hours"] = int((needed > 60 / self.min_headway).sum())
        fixed = self._plan(peak, boardings, np.full_like(peak, 60 / fixed_headway))

        if actual_load is not None:
            a_peak = np.maximum(np.asarray(actual_load, dtype=float), 0).max(axis=0)
            a_board = np.maximum(np.asarray(actual_boardings, dtype=float), 0)
            for plan in (optimized, fixed):
                real = self._plan(a_peak, a_board, plan["departures"])
                plan["realized_overcrowded_hours"] = real["overcrowded_hours"]
                plan["realized_avg_wait_min"] = real["avg_wait_min"]
                plan["realized_max_utilisation"] = real["max_utilisation"]

        return {
            "peak_load": peak,
            "peak_segment": peak_segment,
            "optimized": optimized,
            "fixed": fixed,
            "savings": {
                "cost_pct": (fixed["cost"] - optimized["cost"]) / fixed["cost"] * 100,
                "vehicle_hours": fixed["vehicle_hours"] - optimized["vehicle_hours"],
                "wait_change_min": optimized["avg_wait_min"] - fixed["avg_wait_min"],
                "overcrowding_reduction": fixed["overcrowded_hours"] - optimized["overcrowded_hours"],
            },
        }


def summary_table(result: dict) -> str:
    f, o, s = result["fixed"], result["optimized"], result["savings"]
    rows = [
        ("Vehicle-hours / day", f["vehicle_hours"], o["vehicle_hours"]),
        ("Departures / day", f["departures_total"], o["departures_total"]),
        ("Operating cost", f["cost"], o["cost"]),
        ("Avg wait (min)", f["avg_wait_min"], o["avg_wait_min"]),
        ("Overcrowded hours", f["overcrowded_hours"], o["overcrowded_hours"]),
    ]
    lines = [f"{'':24s}{'Fixed':>10s}{'Optimized':>11s}"]
    lines += [f"{name:24s}{a:>10.1f}{b:>11.1f}" for name, a, b in rows]
    lines.append(f"Cost change: {-s['cost_pct']:+.1f}%   wait change: {s['wait_change_min']:+.1f} min")
    return "\n".join(lines)
