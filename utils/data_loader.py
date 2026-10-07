"""
utils/data_loader.py
--------------------
Loads and preprocesses the UrbanBus dataset.

Filters to ONE route while streaming each raw CSV chunk, so the full
multi-GB files are never held in memory.

- Reads raw AFC transaction CSV files (chunk by chunk)
- Aggregates to 15-minute stop-level boardings
- Adds weather (Open-Meteo), holiday calendar and cyclic time features

The raw CSVs are only needed to rebuild ridership. Weather and calendar
columns of an existing processed dataset can be recomputed with
refresh_exogenous_features() (see scripts/refresh_features.py).
"""

import re
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

AGG_MINUTES = 15
WEATHER_COLS = ["temperature", "precipitation", "wind_speed", "weather_code", "is_raining"]

# Stable integer code per holiday, so the same code means the same holiday in
# every month regardless of which months are processed together.
HOLIDAY_CODES = {
    "None": 0, "New Year's Day": 1, "Chinese New Year": 2, "Tomb-Sweeping Day": 3,
    "Labor Day": 4, "Dragon Boat Festival": 5, "Mid-Autumn Festival": 6,
    "National Day": 7, "Day off": 8,
}

RENAME_MAP = {
    "Bus_Service_Number": "route_id",
    "Boarding_stop_stn":  "stop_id",
    "Ride_start_date":    "date",
    "Ride_start_time":    "time",
}


def month_file(month: str, data_dir: Path = config.RAW_DIR) -> Path:
    return data_dir / f"BUS_DATA_{month.upper()}.csv"


def peek_routes(data_dir: Path = config.RAW_DIR, sample_rows: int = 500_000) -> pd.Series:
    """Transaction counts per route in the first rows of the first raw CSV."""
    csv_files = sorted(data_dir.glob("BUS_DATA_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No BUS_DATA_*.csv files in {data_dir}")
    sample = pd.read_csv(csv_files[0], nrows=sample_rows, usecols=["Bus_Service_Number"])
    counts = sample["Bus_Service_Number"].value_counts()
    print(counts.head(20).to_string())
    return counts


# ── Load one route ────────────────────────────────────────────────────────────
def load_single_route(files: list, target_route: str = config.TARGET_ROUTE,
                      chunk_size: int = config.CHUNK_SIZE) -> pd.DataFrame:
    """Streams the given raw CSVs and keeps only rows of target_route."""
    if not files:
        raise FileNotFoundError(
            f"No raw files given. Download BUS_DATA_*.csv into {config.RAW_DIR} from "
            "https://drive.google.com/drive/folders/1M-zHSNxdp9NfYwu7H3F1uJtsAnyou6rP"
        )
    chunks, total_read = [], 0
    for f in files:
        reader = pd.read_csv(f, chunksize=chunk_size, usecols=list(RENAME_MAP))
        for chunk in tqdm(reader, desc=f"Reading {Path(f).name}", leave=False):
            total_read += len(chunk)
            chunk = chunk[chunk["Bus_Service_Number"] == target_route].rename(columns=RENAME_MAP)
            if chunk.empty:
                continue
            chunk["timestamp"] = pd.to_datetime(
                chunk["date"].astype(str) + " " + chunk["time"].astype(str), errors="coerce")
            chunks.append(chunk[["timestamp", "stop_id"]].dropna())

    if not chunks:
        raise ValueError(f"Route '{target_route}' not found. Run peek_routes() to list routes.")
    df = pd.concat(chunks, ignore_index=True)
    print(f"Route {target_route}: {len(df):,} records kept from {total_read:,} rows")
    return df


def aggregate_ridership(df: pd.DataFrame, slots: pd.DatetimeIndex, stops: list) -> pd.DataFrame:
    """
    15-min boardings per stop on the full (slot × stop) grid, zero where no
    boarding was recorded. The grid is passed in so that every month covers the
    same stops and whole days even when a stop has no boardings in that month.
    """
    df = df.assign(slot=df["timestamp"].dt.floor(f"{AGG_MINUTES}min"))
    agg = df.groupby(["slot", "stop_id"]).size().rename("ridership")
    full_idx = pd.MultiIndex.from_product([slots, stops], names=["slot", "stop_id"])
    agg = agg.reindex(full_idx, fill_value=0).reset_index()
    print(f"Aggregated: {agg.shape}  |  stops: {len(stops)}  |  slots: {len(slots)}")
    return agg


def month_slots(month: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(pd.to_datetime(month, format="%b_%Y"))
    end = start + pd.offsets.MonthBegin(1)
    return pd.date_range(start, end, freq=f"{AGG_MINUTES}min", inclusive="left")


# ── Weather (Open-Meteo archive, no API key) ──────────────────────────────────
def fetch_weather(start: pd.Timestamp, end: pd.Timestamp,
                  lat: float = config.CITY_LAT, lon: float = config.CITY_LON,
                  tz: str = config.CITY_TIMEZONE) -> pd.DataFrame:
    """
    Hourly weather indexed by LOCAL wall-clock time (naive timestamps, same
    convention as the ridership slots).

    The API is asked for UTC and converted here; the Open-Meteo SDK always
    returns epoch seconds, so reading them as local time shifts weather by the
    UTC offset (8 h for China).
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    session = retry(requests_cache.CachedSession(str(config.WEATHER_CACHE), expire_after=-1),
                    retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=session)
    params = {
        "latitude": lat, "longitude": lon,
        # one day of margin on both sides so the local-time window is fully covered
        "start_date": (start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date":   (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "weather_code"],
        "timezone": "GMT",
    }
    hourly = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0].Hourly()
    utc = pd.date_range(pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left")
    weather = pd.DataFrame({
        "slot_hour":     utc.tz_convert(tz).tz_localize(None),
        "temperature":   hourly.Variables(0).ValuesAsNumpy(),
        "precipitation": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed":    hourly.Variables(2).ValuesAsNumpy(),
        "weather_code":  hourly.Variables(3).ValuesAsNumpy(),
    })
    weather["is_raining"] = weather["weather_code"].between(51, 82).astype(int)
    return weather


def merge_weather(agg: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Joins hourly weather onto the slots. Short gaps (≤ 3 h) are interpolated;
    anything longer raises instead of silently becoming 0 °C.
    """
    agg = agg.drop(columns=[c for c in WEATHER_COLS if c in agg.columns])
    hours = pd.DatetimeIndex(agg["slot"].dt.floor("h").unique()).sort_values()
    w = weather.set_index("slot_hour").reindex(hours)
    missing = w["temperature"].isna()
    if missing.any():
        w = w.interpolate(limit=3, limit_direction="both")
        still = w["temperature"].isna()
        if still.any():
            raise ValueError(f"Weather missing for {int(still.sum())} hours, "
                             f"e.g. {list(w.index[still][:3])}")
        print(f"  Weather: interpolated {int(missing.sum())} missing hours")
    w["is_raining"] = (w["is_raining"] > 0.5).astype(int)
    agg["slot_hour"] = agg["slot"].dt.floor("h")
    agg = agg.merge(w, left_on="slot_hour", right_index=True, how="left")
    return agg.drop(columns=["slot_hour"])


# ── Calendar ──────────────────────────────────────────────────────────────────
def _holiday_code(name: str) -> int:
    for key, code in HOLIDAY_CODES.items():
        if key != "None" and name.startswith(key):
            return code
    return HOLIDAY_CODES["Day off"]


def add_holiday_features(agg: pd.DataFrame, country: str = config.CITY_COUNTRY) -> pd.DataFrame:
    """
    is_holiday, holiday_type (stable codes), is_makeup_workday and is_day_off.

    Chinese bridge holidays move a weekend day to a working day
    ("Day off (substituted from 09/30/2017)"); those weekend dates are marked
    as working days.
    """
    dates = agg["slot"].dt.normalize()
    years = sorted(dates.dt.year.unique().tolist())
    cal = holidays.country_holidays(country, years=years + [years[-1] + 1])

    makeup = set()
    for name in cal.values():
        m = re.search(r"substituted from (\d{2}/\d{2}/\d{4})", name)
        if m:
            makeup.add(pd.to_datetime(m.group(1), format="%m/%d/%Y"))

    uniq = pd.DatetimeIndex(dates.unique())
    info = pd.DataFrame(index=uniq)
    info["is_holiday"] = [int(d.date() in cal) for d in uniq]
    info["holiday_type"] = [_holiday_code(cal[d.date()]) if d.date() in cal else 0 for d in uniq]
    info["is_makeup_workday"] = [int(d in makeup) for d in uniq]
    weekend = uniq.dayofweek >= 5
    info["is_day_off"] = ((weekend & (info["is_makeup_workday"] == 0)) | (info["is_holiday"] == 1)).astype(int)

    agg = agg.drop(columns=[c for c in info.columns if c in agg.columns])
    agg["_date"] = dates
    return agg.join(info, on="_date").drop(columns=["_date"])


def add_temporal_features(agg: pd.DataFrame) -> pd.DataFrame:
    """Cyclic time-of-day and day-of-week encodings."""
    slots_per_day = 24 * 60 // AGG_MINUTES
    agg["hour"] = agg["slot"].dt.hour
    agg["minute"] = agg["slot"].dt.minute
    agg["dayofweek"] = agg["slot"].dt.dayofweek
    agg["is_weekend"] = (agg["dayofweek"] >= 5).astype(int)
    agg["month"] = agg["slot"].dt.month
    agg["slot_index"] = agg["hour"] * (60 // AGG_MINUTES) + agg["minute"] // AGG_MINUTES
    agg["sin_time"] = np.sin(2 * np.pi * agg["slot_index"] / slots_per_day)
    agg["cos_time"] = np.cos(2 * np.pi * agg["slot_index"] / slots_per_day)
    agg["sin_dow"] = np.sin(2 * np.pi * agg["dayofweek"] / 7)
    agg["cos_dow"] = np.cos(2 * np.pi * agg["dayofweek"] / 7)
    return agg


def refresh_exogenous_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recomputes weather, holiday and time columns for an existing dataset."""
    df = df.copy()
    df["slot"] = pd.to_datetime(df["slot"])
    weather = fetch_weather(df["slot"].min(), df["slot"].max())
    df = merge_weather(df, weather)
    df = add_holiday_features(df)
    return add_temporal_features(df)


# ── POI features (OpenStreetMap Overpass) ─────────────────────────────────────
def fetch_poi_features(stop_df: pd.DataFrame, radius_m: int = 500) -> pd.DataFrame:
    """
    Counts POIs within radius_m of each stop. stop_df needs stop_id, lat, lon.
    Query errors are reported and left as NaN rather than counted as 0.
    """
    import overpy
    api = overpy.Overpass()
    tags = {
        "school":   'node["amenity"="school"]',
        "hospital": 'node["amenity"="hospital"]',
        "mall":     'node["shop"="mall"]',
        "stadium":  'node["leisure"="stadium"]',
        "park":     'node["leisure"="park"]',
    }
    rows = []
    for _, r in tqdm(stop_df.iterrows(), total=len(stop_df), desc="Fetching POIs"):
        counts = {"stop_id": r["stop_id"]}
        for name, tag in tags.items():
            query = f"[out:json][timeout:25];{tag}(around:{radius_m},{r['lat']},{r['lon']});out ids;"
            try:
                counts[f"poi_{name}"] = len(api.query(query).nodes)
            except Exception as e:  # network / rate limit
                print(f"  POI query failed for {r['stop_id']} {name}: {e}")
                counts[f"poi_{name}"] = np.nan
        rows.append(counts)
    poi = pd.DataFrame(rows)
    poi["poi_total"] = poi.filter(like="poi_").sum(axis=1, min_count=1)
    return poi


def load_stop_list(data_dir: Path = config.RAW_DIR):
    """BusStopList.csv with stop_id/lat/lon columns, or None if unavailable."""
    path = data_dir / "BusStopList.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"bus_stop": "stop_id", "latitude": "lat", "longitude": "lon"})
    if not {"stop_id", "lat", "lon"} <= set(df.columns):
        print(f"BusStopList.csv has no coordinates ({list(df.columns)}) — POI features skipped")
        return None
    return df


# ── Master pipeline ───────────────────────────────────────────────────────────
def build_month(month: str, stops: list = None) -> pd.DataFrame:
    """Processes one raw month into the 15-min feature table."""
    path = month_file(month)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — download it into {config.RAW_DIR}")
    raw = load_single_route([path])
    stops = stops or sorted(set(config.ROUTE_STOP_ORDER) | set(raw["stop_id"].unique()))
    agg = aggregate_ridership(raw, month_slots(month), stops)
    agg = merge_weather(agg, fetch_weather(agg["slot"].min(), agg["slot"].max()))
    agg = add_holiday_features(agg)
    return add_temporal_features(agg)


def build_dataset(months: list = config.MONTHS, use_poi: bool = False) -> pd.DataFrame:
    """Builds and saves the full processed dataset from the raw monthly CSVs."""
    df = pd.concat([build_month(m) for m in months], ignore_index=True)
    if use_poi:
        stop_df = load_stop_list()
        if stop_df is not None:
            df = df.merge(fetch_poi_features(stop_df), on="stop_id", how="left")
    save_parquet_atomic(df, config.DATASET_PATH)
    return df


def save_parquet_atomic(df: pd.DataFrame, path: Path):
    """Writes to a temp file and renames, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    print(f"Saved {path}  |  shape {df.shape}")


if __name__ == "__main__":
    build_dataset()
