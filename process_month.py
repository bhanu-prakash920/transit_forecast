"""
process_month.py
----------------
Processes one raw month and merges it into data/processed/dataset_all.parquet,
so only one raw CSV (~17-20 GB) needs to be on disk at a time.

Usage:
    python process_month.py OCT_2017
    python process_month.py status

Re-processing a month replaces that month's rows. After a successful run the
raw CSV can be deleted.
"""

import sys

import pandas as pd

import config
from utils.data_loader import build_month, save_parquet_atomic


def process_single_month(month: str) -> bool:
    month = month.upper()
    if month not in config.MONTHS:
        print(f"Unknown month {month}. Available: {', '.join(config.MONTHS)}")
        return False
    try:
        new = build_month(month)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False

    if config.DATASET_PATH.exists():
        existing = pd.read_parquet(config.DATASET_PATH)
        existing["slot"] = pd.to_datetime(existing["slot"])
        # Replace any rows of the same month instead of keeping stale ones.
        in_month = existing["slot"].between(new["slot"].min(), new["slot"].max())
        existing = existing[~in_month]
        if set(existing.columns) != set(new.columns):
            print(f"⚠ Column mismatch — existing only: {set(existing.columns) - set(new.columns)}, "
                  f"new only: {set(new.columns) - set(existing.columns)}. "
                  "Re-run scripts/refresh_features.py after all months are processed.")
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    # Every stop must appear in every slot, otherwise the model sees gaps.
    stops = sorted(combined["stop_id"].unique())
    slots = pd.DatetimeIndex(sorted(combined["slot"].unique()))
    combined = (combined.set_index(["slot", "stop_id"])
                        .reindex(pd.MultiIndex.from_product([slots, stops], names=["slot", "stop_id"]))
                        .reset_index())
    missing = combined["ridership"].isna()
    if missing.any():
        print(f"⚠ {int(missing.sum()):,} slot×stop rows had no record (stop absent that month) — "
              "filled with 0 boardings; run scripts/refresh_features.py to fill their features.")
        combined["ridership"] = combined["ridership"].fillna(0)
    combined["ridership"] = combined["ridership"].astype(int)

    save_parquet_atomic(combined.sort_values(["stop_id", "slot"]), config.DATASET_PATH)
    print(f"✅ {month} merged. You can now delete {config.RAW_DIR / f'BUS_DATA_{month}.csv'}")
    return True


def show_status():
    if not config.DATASET_PATH.exists():
        print("No processed data found yet.")
        return
    df = pd.read_parquet(config.DATASET_PATH, columns=["slot", "stop_id"])
    months = sorted(pd.to_datetime(df["slot"]).dt.strftime("%b_%Y").str.upper().unique())
    print(f"Rows: {len(df):,} | Stops: {df['stop_id'].nunique()} | "
          f"Range: {df['slot'].min()} → {df['slot'].max()}")
    missing = [m for m in config.MONTHS if m not in months]
    print("Missing months: " + (", ".join(missing) if missing else "none ✅"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        show_status()
        sys.exit(1)
    if sys.argv[1].lower() == "status":
        show_status()
    else:
        sys.exit(0 if process_single_month(sys.argv[1]) else 1)
