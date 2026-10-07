"""
config.py
---------
Single source of truth for paths, route definition, and experiment settings.

All paths are absolute (resolved from this file), so scripts, notebooks and the
API work no matter which directory they are launched from.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
OUTPUT_DIR     = ROOT_DIR / "outputs"
ARTIFACT_DIR   = OUTPUT_DIR / "artifacts"      # model weights + metadata used by the API
RESULTS_DIR    = OUTPUT_DIR / "results"        # metric tables, predictions
PLOT_DIR       = OUTPUT_DIR / "plots"
DASHBOARD_DIR  = ROOT_DIR / "dashboard"
WEATHER_CACHE  = DATA_DIR / "weather_cache"    # requests-cache sqlite (".sqlite" is appended)

DATASET_PATH   = PROCESSED_DIR / "dataset_all.parquet"
OD_TRIPS_PATH  = PROCESSED_DIR / "od_trips.parquet"

# ── Route ─────────────────────────────────────────────────────────────────────
TARGET_ROUTE = "SER_dccb"
CHUNK_SIZE   = 500_000   # rows per chunk when streaming the raw CSVs

# Physical stop sequence of the route. Derived from the OD trips with
# utils.route.infer_stop_order(): 92% of trips travel "forward" in this order and
# almost all of the rest alight at EV_2 after passing the I_4 terminus, i.e. the
# route is a loop (I_4 → BUSPARK → EV_2 → ...). Stop IDs are not in route order
# alphabetically, so never use sorted(stop_ids) where the physical order matters.
ROUTE_STOP_ORDER = [
    "BUSPARK", "EV_2", "MH_2", "ELY_1", "PL_2", "QN_1", "AXF_1", "AVX_1",
    "BWK_1", "BXK_1", "EVJ_1", "JB_1", "ATK_1", "BYQ_1", "SL_1", "DJJ_1",
    "GS_2", "CQG_1", "BOR_1", "HT_2", "F_5", "I_4",
]
ROUTE_IS_LOOP = True

# Months available in the UrbanBus dataset (raw file suffixes).
MONTHS = ["OCT_2017", "NOV_2017", "DEC_2017", "JAN_2018", "FEB_2018", "MAR_2018"]

# ── Location (weather + holiday calendar) ─────────────────────────────────────
# The original pipeline assumed Shenzhen. The ridership shows no Golden Week or
# Spring Festival effect, so the city is unverified; weather and holiday
# features are kept but their value is measured by the feature ablation.
CITY_LAT      = 22.5431
CITY_LON      = 114.0579
CITY_TIMEZONE = "Asia/Shanghai"
CITY_COUNTRY  = "CN"

# ── Dataset / forecasting setup ───────────────────────────────────────────────
LOOKBACK_HOURS = 48
FORECAST_HOURS = 24

# Chronological split boundaries (inclusive start of val / test).
# 2017-10-01 .. 2018-01-31 train (4 months), Feb val, Mar test.
VAL_START  = "2018-02-01"
TEST_START = "2018-03-01"

# Stops whose mean hourly boardings in the TRAINING period fall below this are
# not forecast by the neural models (they are near-constant zero). Their
# boardings are still handled by the optimizer via historical averages.
MIN_STOP_RIDERSHIP = 2.0

SEEDS = [0, 1, 2]

TRAIN = dict(
    batch_size   = 32,
    lr           = 1e-3,
    weight_decay = 1e-4,
    max_epochs   = 60,
    patience     = 10,
    hidden_dim   = 64,
    dropout      = 0.1,
)

# ── Optimizer / operations ────────────────────────────────────────────────────
BUS_CAPACITY        = 80      # passengers per bus
FIXED_HEADWAY_MIN   = 15      # current fixed schedule
MIN_HEADWAY_MIN     = 5
MAX_HEADWAY_MIN     = 30
ROUND_TRIP_MIN      = 60      # assumed cycle time of one loop incl. layover (not in the data)
COST_PER_VEH_HOUR   = 50      # currency units per vehicle-hour (assumed)
# Plan buses for this share of capacity so forecast errors do not overcrowd
# them. Chosen on the validation month with scripts/evaluate_schedule.py --tune.
PLANNING_LOAD_FACTOR = 0.85
