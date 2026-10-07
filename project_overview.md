# Project overview

## Problem

Buses on route `SER_dccb` run on a fixed timetable (every 15 minutes, all day). Demand is not
flat: in the test month the fixed schedule was overcrowded on the busiest segment for 251
hours, on every single day, while night buses ran nearly empty. The goal is to forecast
tomorrow's boardings at every stop and hour, convert them into passengers per route segment,
and plan how many buses to run each hour.

## Data

- UrbanBus smart-card data, October 2017 – March 2018, one route: 22 stops, 840,729 OD trips
  in Oct–Nov (boarding and alighting stop), boardings for all six months.
- Weather from the Open-Meteo archive and the Chinese holiday calendar. Both assume Shenzhen.
  The ridership shows no Golden Week or Spring Festival dip and Saturday is the busiest day,
  so the location is uncertain, and neither feature improves accuracy (see ablation).

## Method

1. **Features** (per stop and hour): last 48 h of boardings, calendar, weather, boardings a
   week earlier; for every target hour, the known-ahead calendar, boardings 24 h / 168 h
   earlier, and the stop's training-period hour-of-week average.
2. **Models**: baselines (seasonal naive, hour-of-week mean), direct XGBoost, and graph neural
   networks (GRU+GAT on the route graph and on an OD-flow graph, BiLSTM with multi-scale
   attention, AGCRN with a learned graph). Neural decoders start from the hour-of-week profile
   and learn corrections to it.
3. **Ensemble**: weights and a residual XGBoost fit on the validation month only.
4. **Route and loads**: stop order recovered from the OD trips (the route is a loop);
   forecast boardings are sent to destinations with the historical P(destination | stop, hour)
   and routed along the loop.
5. **Frequency plan**: each hour, enough departures to carry the busiest segment at 85% of
   capacity (the buffer is chosen on the validation month), headways 5–30 min.

Evaluation: train Oct–Jan, validation February, test March. Every number below is on March,
which nothing was tuned on.

## Results

| | Test WMAPE |
|---|---|
| Ensemble + residual XGBoost | **18.80%** (MAE 5.89 passengers per stop-hour) |
| XGBoost | 18.95% |
| Graph neural networks (all variants) | 19.7–20.0% |
| Hour-of-week average | 19.97% |
| Same hour last week | 24.11% |

Frequency plan vs the fixed 15-min timetable, scored against actual March demand: overcrowded
hours 251 → 8, average wait 9.3 → 6.0 min, at 12.8% more vehicle-hours. With perfect
forecasts the same result would cost nothing extra; the 12.8% is the cost of forecast error.

## What changed from the earlier version, and why the numbers moved

The earlier version reported 20.11% WMAPE and "12.5% cheaper with zero overcrowding". Those
numbers came from a pipeline with several faults:

| Fault | Effect | Fix |
|---|---|---|
| Lag features of one stop (ATK_1) copied to all stops | wrong inputs for 16 of 17 stops | per-stop feature arrays |
| Scalers, `peak_ratio` and stop filter fit on all data incl. test | optimistic test scores | fit on training period only |
| Stops dropped *because* their error was high | optimistic test scores | fixed threshold on training ridership |
| Overlapping train/val/test targets, model chosen on test | optimistic test scores | date-based splits, validation-only selection |
| No naive baseline | the old "best" model was no better than the hour-of-week average | baselines added |
| Weather 8 h out of step, zero-filled gaps | noise features | UTC → local conversion, gaps interpolated |
| Edge weights ignored by GAT; near-complete distance graph | "spatial" part did nothing | edge-weighted GAT, route and OD graphs |
| Stops processed in alphabetical order in the optimizer | loads and bottlenecks between non-adjacent stops | stop order recovered from OD trips (loop route) |
| Overcrowding judged against the forecast itself | optimized plan was 0 by construction | plans scored against actual demand |
| Dashboard figures hand-entered or randomly generated | not reproducible | computed from saved results and models |

## Limitations

One route, one test month, uncertain location, and operating assumptions (capacity 80, 60-min
round trip, cost per vehicle-hour) that are not in the data. Future weather is not used.
