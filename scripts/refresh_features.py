"""
scripts/refresh_features.py
---------------------------
Recomputes the weather, holiday and time columns of the existing processed
dataset without needing the raw CSVs (ridership is kept as is).

Fixes datasets produced by the old pipeline, whose weather was shifted by the
UTC offset and zero-filled on the last day of every month.

    python scripts/refresh_features.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from utils.data_loader import refresh_exogenous_features, save_parquet_atomic  # noqa: E402


def main():
    df = pd.read_parquet(config.DATASET_PATH)
    before = df.copy()
    df = refresh_exogenous_features(df)
    df = df.sort_values(["stop_id", "slot"]).reset_index(drop=True)

    b = before.sort_values(["stop_id", "slot"]).reset_index(drop=True)
    assert (b["ridership"].values == df["ridership"].values).all(), "ridership changed"

    hourly = df[df["stop_id"] == df["stop_id"].iloc[0]].groupby(df["slot"].dt.hour)["temperature"].mean()
    print(f"Warmest hour of day: {hourly.idxmax()}:00 (expect ~14:00 local), "
          f"coldest: {hourly.idxmin()}:00")
    print(f"Rows with temperature == 0: {(df['temperature'] == 0).sum()} (was {(b['temperature'] == 0).sum()})")
    save_parquet_atomic(df, config.DATASET_PATH)


if __name__ == "__main__":
    main()
