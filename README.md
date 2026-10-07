# Transit demand forecasting & load-aware frequency planning

Day-ahead forecasts of hourly boardings at every stop of UrbanBus route `SER_dccb`,
turned into passenger loads per route segment and a bus frequency plan.

```
smart-card boardings ─┐
weather, calendar ────┼─► 48 h history per stop ─┐
                      │   + known-ahead features ├─► models ─► ensemble ─► boardings per stop/hour
OD trips (Oct–Nov) ───┘                          │                          │
                                                 │   P(destination | stop, hour)
                                                 └──────────────────────────┴─► segment loads ─► departures per hour
```

## Results

Test month: March 2018 (31 days, 721 hourly forecast origins, 19 stops). It was never used for
training, early stopping, tuning or ensemble weights. Neural models and XGBoost: mean ± std
over 3 seeds. Errors are in boardings per stop-hour.

| Model | MAE | RMSE | WMAPE % | MAPE % (actual ≥ 10) |
|---|---|---|---|---|
| **Ensemble + residual XGBoost** | **5.89** | 11.07 | **18.80** | **21.72** |
| Ensemble (weighted) | 5.91 | **10.93** | 18.85 | 21.77 |
| XGBoost (direct) | 5.94 ± 0.04 | 11.10 ± 0.11 | 18.95 ± 0.11 | 21.85 ± 0.01 |
| AGCRN (learned graph) | 6.17 ± 0.02 | 11.36 ± 0.09 | 19.69 ± 0.08 | 22.63 ± 0.04 |
| STGNN (route graph) | 6.17 ± 0.03 | 11.41 ± 0.09 | 19.70 ± 0.08 | 22.61 ± 0.07 |
| STGNN (OD-flow graph) | 6.17 ± 0.03 | 11.40 ± 0.10 | 19.70 ± 0.08 | 22.62 ± 0.08 |
| GRU (no graph) | 6.17 ± 0.03 | 11.41 ± 0.10 | 19.70 ± 0.10 | 22.57 ± 0.06 |
| Historical hour-of-week mean | 6.26 | 11.58 | 19.97 | 22.72 |
| STGNN-v3 (BiLSTM + attention) | 6.26 ± 0.01 | 11.65 ± 0.02 | 19.98 ± 0.02 | 22.60 ± 0.12 |
| Seasonal naive (last week) | 7.55 | 13.84 | 24.11 | 29.53 |
| Seasonal naive (yesterday) | 9.18 | 18.42 | 29.29 | 33.66 |

Forecasts issued only at midnight (true day-ahead, 31 forecasts) give the same ranking; the
ensemble scores 18.82% WMAPE (`outputs/results/metrics_dayahead.csv`).

What the numbers say:

- The **hour-of-week average is a strong baseline** (19.97%). The neural models beat it by
  about 0.3 points, XGBoost by 1 point, and the ensemble by 1.2 points.
- **The graph adds nothing measurable here.** GRU without a graph, STGNN on the route graph,
  STGNN on the OD-flow graph and AGCRN all land at 19.7% ± 0.1. With 19 stops on one route, a
  stop's own history and weekly profile explain almost everything the neighbours could.
- The ensemble weights (fit on February) are 0.53 XGBoost and 0.47 STGNN (route graph).

Feature ablation (GRU without a graph, 3 seeds, test WMAPE %):

| Variant | WMAPE % |
|---|---|
| all features | 19.70 ± 0.10 |
| without weather | 19.76 ± 0.12 |
| without holiday / day-off flags | 19.69 ± 0.08 |
| without hour-of-week profile | 21.36 ± 0.32 |

Weather and holiday flags make no difference, which fits the doubts about the assumed location
(see Limitations).

**Frequency plan**, test month, scored against the demand that actually happened
(capacity 80, fixed schedule every 15 min, assumed 60-min round trip):

| Plan | Vehicle-hours / day | Overcrowded hours (month) | Days with overcrowding | Avg wait (min) |
|---|---|---|---|---|
| Fixed 15-min headway | 96.0 | 251 | 31 of 31 | 9.3 |
| Forecast-driven, no buffer | 94.8 (−1.2%) | 110 | 28 | 7.1 |
| **Forecast-driven, 15% buffer** | 108.3 (+12.8%) | **8** | 4 | **6.0** |
| Oracle (plans on actual demand) | 95.9 (−0.1%) | 0 | 0 | 6.8 |

The fixed schedule is not too large but badly distributed across the day: with perfect
knowledge the same number of vehicle-hours would remove all overcrowding. Forecast error is
what costs the extra 12.8%. The buffer (plan to 85% of capacity) was chosen on the
validation month.

## Quick start

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

python scripts/run_experiments.py          # train + evaluate everything (~1–2 h on an M-series Mac)
python scripts/evaluate_schedule.py        # score the frequency plans on the test month
python dashboard/data_export.py            # build dashboard/data.json from the results
python -m backend.app                      # dashboard + API on http://127.0.0.1:8000
pytest                                     # unit tests
```

`python scripts/run_experiments.py --quick` runs the whole pipeline with one seed and three
epochs (a few minutes) to check that everything works.

## Data

| File | Needed for | How to get it |
|------|------------|---------------|
| `data/processed/dataset_all.parquet` | everything | built from the raw CSVs (below) |
| `data/processed/od_trips.parquet` | OD graph, segment loads | `python -m utils.od_matrix OCT_2017 NOV_2017` |
| `data/raw/BUS_DATA_<MON>_<YEAR>.csv` | rebuilding the two files above | [UrbanBus Google Drive](https://drive.google.com/drive/folders/1M-zHSNxdp9NfYwu7H3F1uJtsAnyou6rP) |

The raw monthly CSVs are 10–20 GB each and only needed to rebuild ridership. Process one at a
time and delete it afterwards:

```bash
python process_month.py OCT_2017      # repeat for each month; `python process_month.py status`
```

Weather and calendar columns can be recomputed without the raw files:
`python scripts/refresh_features.py`.

## How it works

**Setup.** Hourly boardings for the 19 stops that average ≥ 2 boardings/hour in the training
period (3 near-empty stops are left to historical averages). Chronological split:
train 2017-10-01 → 2018-01-31, validation February 2018, test March 2018. A sample is a
forecast origin; all 24 target hours of a sample lie inside one split, so no target is ever
shared between splits. Scalers, the stop filter and the hour-of-week profile are fit on
training hours only. The test month is used for nothing but the final numbers.

**Inputs.** Per stop: the last 48 h of boardings, calendar (hour, weekday, day off, holiday),
weather, and boardings one week earlier. For each target hour, the model also gets what is
known when the forecast is made: calendar, boardings 24 h and 168 h before, and the stop's
training-period hour-of-week average.

**Models** (`models/`): seasonal naive (yesterday, last week), hour-of-week average,
direct multi-horizon XGBoost, GRU without a graph, STGNN (GRU + edge-weighted GAT) on the
route graph and on an OD-flow graph, STGNN-v3 (BiLSTM + multi-scale attention + GAT) and
AGCRN (learned adjacency, node-adaptive weights). Neural models are trained with an L1 loss
on the passenger scale (equivalent to optimising WMAPE), early-stopped on validation WMAPE,
and repeated over 3 seeds. The ensemble weights and an optional residual XGBoost are fit on
the validation month only.

**Route and loads** (`utils/route.py`). The stop sequence is recovered from the OD trips:
92% of trips run forward in the order in `config.ROUTE_STOP_ORDER` and almost all others
alight at EV_2 after passing the I_4 terminus, so the route is modelled as a loop.
Forecast boardings are split over destinations with the historical P(destination | origin,
hour) and routed along the loop to get passengers per segment.

**Frequency plan** (`optimize/frequency.py`). Each hour, enough departures that the busiest
segment fits in the buses at a planning load factor (0.85, chosen on the validation month),
with headways between 5 and 30 min. Plans are scored against the demand that actually
occurred.

## Project layout

```
config.py                 paths, route, split dates, hyper-parameters, operating assumptions
process_month.py          raw month → processed dataset
utils/data_loader.py      raw CSV streaming, weather, holidays, POIs
utils/od_matrix.py        OD trip extraction, boarding/alighting, segment loads
utils/dataset.py          leakage-free windows and features
utils/route.py            stop-order inference, OD load model
utils/graph_builder.py    route, OD-flow and distance graphs
utils/metrics.py          MAE, RMSE, WMAPE, MAPE(y ≥ 10)
models/stgnn.py           STGNN, STGNN-v3, AGCRN
models/baselines.py       naive, hour-of-week mean, XGBoost
models/ensemble.py        validation-fit weights, residual corrector
models/trainer.py         training loop
models/forecaster.py      model registry, saving/loading the ensemble
models/evaluate.py        figures
scripts/                  run_experiments, evaluate_schedule, refresh_features
backend/                  FastAPI app + inference engine
dashboard/                static dashboard (data.json + live /api/simulate)
notebooks/main.ipynb      walkthrough of data and results
tests/                    pytest suite
```

## Limitations

- **Location is uncertain.** The code assumes Shenzhen for weather and the Chinese holiday
  calendar, but ridership shows no Golden Week or Spring Festival effect and Saturday is the
  busiest day. The feature ablation shows how much weather and holiday flags contribute.
- **One route, six months, one test month.** The test month (31 days) is small; differences of
  a few tenths of a WMAPE point between models are within seed noise.
- **Operating assumptions are not in the data**: bus capacity (80), round-trip time (60 min)
  and cost per vehicle-hour (50). They change the cost numbers, not the forecasts.
- **Alightings use the ride start hour.** The data has no alighting time.
- **Future weather is not used** (a real deployment would need a weather forecast).
- **POI features are off**: `BusStopList.csv` has no coordinates and the distance matrix is not
  available locally, so the distance graph is implemented but untested on real data.
