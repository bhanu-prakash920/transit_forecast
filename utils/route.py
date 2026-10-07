"""
utils/route.py
--------------
Route geometry helpers: stop order inference and passenger load on segments.

Load model
----------
Forecasts give boardings per stop and hour. To get the number of passengers on
the bus between consecutive stops we need to know where those passengers get
off. We estimate P(destination | origin, hour) from historical OD trips and
route each expected trip along the stop sequence. On a loop route, a trip whose
destination comes before its origin wraps around the end of the sequence.
"""

import numpy as np
import pandas as pd


def od_count_matrix(trips: pd.DataFrame, stops: list) -> np.ndarray:
    """[n, n] trip counts, rows = boarding stop, cols = alighting stop."""
    c = (trips.groupby(["boarding_stop", "alighting_stop"]).size()
              .unstack(fill_value=0)
              .reindex(index=stops, columns=stops, fill_value=0))
    return c.values.astype(float)


def infer_stop_order(trips: pd.DataFrame) -> tuple[list, float]:
    """
    Orders stops so that as many trips as possible travel forward.

    Greedy Copeland ranking followed by single-stop insertion moves until no
    move improves the forward-trip count. Returns (order, forward_fraction).
    """
    stops = sorted(set(trips["boarding_stop"]) | set(trips["alighting_stop"]))
    c = od_count_matrix(trips, stops)
    np.fill_diagonal(c, 0)
    n = len(stops)

    def forward(p):
        return np.triu(c[np.ix_(p, p)], 1).sum()

    order = list(np.argsort(-(c > c.T).sum(axis=1), kind="stable"))
    best = forward(order)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = order.copy()
                cand.insert(j, cand.pop(i))
                s = forward(cand)
                if s > best:
                    order, best, improved = cand, s, True
    return [stops[i] for i in order], float(best / max(c.sum(), 1))


def segment_names(stop_order: list, is_loop: bool) -> list:
    names = [f"{a}→{b}" for a, b in zip(stop_order[:-1], stop_order[1:])]
    if is_loop:
        names.append(f"{stop_order[-1]}→{stop_order[0]}")
    return names


def segment_incidence(stop_order: list, is_loop: bool) -> np.ndarray:
    """
    inc[o, d, k] = 1 if a passenger boarding at stop o and alighting at stop d
    rides over segment k (segment k joins stop k and k+1, wrapping on a loop).
    Trips with d before o are dropped on a linear route.
    """
    n = len(stop_order)
    n_seg = n if is_loop else n - 1
    inc = np.zeros((n, n, n_seg), dtype=np.float32)
    for o in range(n):
        for d in range(n):
            if o == d:
                continue
            if d > o:
                inc[o, d, o:d] = 1
            elif is_loop:
                inc[o, d, o:] = 1
                inc[o, d, :d] = 1
    return inc


class ODLoadModel:
    """
    Converts boardings per (stop, hour) into alightings and segment loads using
    historical destination shares P(dest | origin, hour-of-day).

    Fit on training-period trips only.
    """

    def __init__(self, stop_order: list, is_loop: bool, min_trips: int = 30):
        self.stop_order = list(stop_order)
        self.is_loop = is_loop
        self.min_trips = min_trips
        self.idx = {s: i for i, s in enumerate(self.stop_order)}
        self.inc = segment_incidence(self.stop_order, is_loop)
        self.segments = segment_names(self.stop_order, is_loop)
        self.dest_share = None   # [24, n, n]

    def fit(self, trips: pd.DataFrame) -> "ODLoadModel":
        n = len(self.stop_order)
        t = trips[trips["boarding_stop"].isin(self.idx) & trips["alighting_stop"].isin(self.idx)]
        t = t[t["boarding_stop"] != t["alighting_stop"]]
        o = t["boarding_stop"].map(self.idx).to_numpy()
        d = t["alighting_stop"].map(self.idx).to_numpy()
        h = pd.to_datetime(t["slot"]).dt.hour.to_numpy()

        counts = np.zeros((24, n, n))
        np.add.at(counts, (h, o, d), 1)
        overall = counts.sum(axis=0)                                    # [n, n]
        overall_share = overall / np.maximum(overall.sum(axis=1, keepdims=True), 1)

        per_hour_tot = counts.sum(axis=2, keepdims=True)                # [24, n, 1]
        share = counts / np.maximum(per_hour_tot, 1)
        # Too few trips for an (origin, hour) → fall back to the origin's all-day mix.
        sparse = per_hour_tot[..., 0] < self.min_trips
        share[sparse] = np.broadcast_to(overall_share, share.shape)[sparse]
        self.dest_share = share
        return self

    def od_flows(self, boarding: np.ndarray, hours: list) -> np.ndarray:
        """boarding [n_stops, T] in route order → expected OD flows [T, n, n]."""
        assert self.dest_share is not None, "call fit() first"
        share = self.dest_share[np.asarray(hours) % 24]                 # [T, n, n]
        return share * boarding.T[:, :, None]

    def alighting(self, boarding: np.ndarray, hours: list) -> np.ndarray:
        """Expected alightings [n_stops, T]."""
        return self.od_flows(boarding, hours).sum(axis=1).T

    def segment_load(self, boarding: np.ndarray, hours: list) -> np.ndarray:
        """Passengers carried over each segment during each hour, [n_seg, T]."""
        flows = self.od_flows(boarding, hours)                          # [T, n, n]
        return np.einsum("tod,odk->kt", flows, self.inc)
