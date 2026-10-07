"""
utils/od_matrix.py
------------------
Extracts origin-destination trips for one route from the raw UrbanBus CSVs
(streamed in chunks) and derives boarding/alighting counts and segment loads.

The raw data only has the ride START time, so each trip is attributed to the
hour in which it started, including its alighting.

Usage:
    python -m utils.od_matrix OCT_2017 NOV_2017
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from utils.data_loader import month_file, save_parquet_atomic  # noqa: E402
from utils.route import segment_incidence, segment_names  # noqa: E402

COLS = ["Bus_Service_Number", "Boarding_stop_stn", "Alighting_stop_stn",
        "Ride_start_date", "Ride_start_time"]


def extract_trips(csv_path, route: str = config.TARGET_ROUTE,
                  chunk_size: int = config.CHUNK_SIZE) -> pd.DataFrame:
    """Trips of one route: boarding_stop, alighting_stop, slot (hour of ride start)."""
    parts = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=COLS):
        c = chunk[chunk["Bus_Service_Number"] == route]
        if c.empty:
            continue
        slot = pd.to_datetime(c["Ride_start_date"].astype(str) + " " + c["Ride_start_time"].astype(str),
                              errors="coerce").dt.floor("h")
        parts.append(pd.DataFrame({"boarding_stop": c["Boarding_stop_stn"].values,
                                   "alighting_stop": c["Alighting_stop_stn"].values,
                                   "slot": slot.values}).dropna())
    trips = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["boarding_stop", "alighting_stop", "slot"])
    print(f"{Path(csv_path).name}: {len(trips):,} trips on route {route}")
    return trips


def boarding_alighting(trips: pd.DataFrame) -> pd.DataFrame:
    """Hourly boardings and alightings per stop."""
    b = trips.groupby(["boarding_stop", "slot"]).size().rename("boarding")
    a = trips.groupby(["alighting_stop", "slot"]).size().rename("alighting")
    b.index.names = a.index.names = ["stop_id", "slot"]
    return pd.concat([b, a], axis=1).fillna(0).astype(int).reset_index()


def load_profile(trips: pd.DataFrame, stop_order: list = config.ROUTE_STOP_ORDER,
                 is_loop: bool = config.ROUTE_IS_LOOP) -> pd.DataFrame:
    """Passengers crossing each route segment per hour (long format)."""
    idx = {s: i for i, s in enumerate(stop_order)}
    t = trips[trips["boarding_stop"].isin(idx) & trips["alighting_stop"].isin(idx)]
    slots = pd.DatetimeIndex(sorted(t["slot"].unique()))
    s_idx = slots.get_indexer(t["slot"])
    n = len(stop_order)
    od = np.zeros((len(slots), n, n))
    np.add.at(od, (s_idx, t["boarding_stop"].map(idx).values, t["alighting_stop"].map(idx).values), 1)
    load = np.einsum("tod,odk->tk", od, segment_incidence(stop_order, is_loop))
    names = segment_names(stop_order, is_loop)
    return (pd.DataFrame(load, index=slots, columns=names)
              .rename_axis("slot").reset_index()
              .melt(id_vars="slot", var_name="segment", value_name="load"))


def process_months(months: list):
    """Extracts trips for the given months and writes/merges od_trips.parquet."""
    new = pd.concat([extract_trips(month_file(m)) for m in months], ignore_index=True)
    if config.OD_TRIPS_PATH.exists():
        old = pd.read_parquet(config.OD_TRIPS_PATH)
        old = old[~old["slot"].dt.to_period("M").isin(new["slot"].dt.to_period("M").unique())]
        new = pd.concat([old, new], ignore_index=True)
    save_parquet_atomic(new, config.OD_TRIPS_PATH)
    save_parquet_atomic(boarding_alighting(new), config.PROCESSED_DIR / "boarding_alighting.parquet")
    return new


if __name__ == "__main__":
    process_months(sys.argv[1:] or config.MONTHS)
