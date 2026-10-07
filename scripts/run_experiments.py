"""
scripts/run_experiments.py
--------------------------
Trains every model, evaluates on the held-out test month, builds the ensemble
and saves the artifacts the API serves.

    python scripts/run_experiments.py                 # all models, 3 seeds
    python scripts/run_experiments.py --quick         # 1 seed, few epochs (smoke test)
    python scripts/run_experiments.py --models gru_nograph stgnn_route --seeds 0

Outputs
  outputs/results/metrics.csv          test metrics per model (mean ± std over seeds)
  outputs/results/metrics_dayahead.csv same, forecasts issued at midnight only
  outputs/results/ablation.csv         feature ablation (if --ablation)
  outputs/results/summary.json         everything above + ensemble weights
  outputs/results/test_predictions.npz ensemble forecasts for the test month
  outputs/artifacts/                   weights + metadata for the API
  outputs/plots/                       figures
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from models import evaluate as ev  # noqa: E402
from models.baselines import XGBoostDirect, hour_of_week_mean, seasonal_naive  # noqa: E402
from models.ensemble import ResidualCorrector, combine, fit_weights  # noqa: E402
from models.forecaster import DISPLAY, NEURAL_SPECS, build_graphs, make_model, neural_predict  # noqa: E402
from models.trainer import Trainer, get_device, set_seed  # noqa: E402
from utils.dataset import get_dataloaders, prepare_data  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(NEURAL_SPECS), choices=list(NEURAL_SPECS))
    p.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    p.add_argument("--epochs", type=int, default=config.TRAIN["max_epochs"])
    p.add_argument("--patience", type=int, default=config.TRAIN["patience"])
    p.add_argument("--quick", action="store_true", help="1 seed, 3 epochs: pipeline smoke test")
    p.add_argument("--ablation", action="store_true", help="also run the feature ablation")
    return p.parse_args()


def fmt(mean, std):
    return f"{mean:.2f} ± {std:.2f}" if std > 0 else f"{mean:.2f}"


def train_member(member, data, graphs, seed, args, device):
    set_seed(seed)
    loaders = get_dataloaders(data, seed=seed)
    graph = graphs.get(NEURAL_SPECS[member][2]) if NEURAL_SPECS[member][2] else None
    model = make_model(member, data)
    trainer = Trainer(model, data.scaler, *(graph or (None, None)),
                      lr=config.TRAIN["lr"], weight_decay=config.TRAIN["weight_decay"],
                      max_epochs=args.epochs, patience=args.patience, device=device, verbose=False)
    t0 = time.time()
    history = trainer.fit(loaders["train"], loaders["val"])
    print(f"  {member} seed {seed}: best epoch {trainer.best_epoch}, "
          f"val WMAPE {min(history['val_wmape']):.2f}%, {time.time() - t0:.0f}s")
    return model, graph, history


def run_ablation(df, args, device):
    """GRU (no graph) with feature groups removed, seed-averaged."""
    rows = []
    variants = {
        "all features": [],
        "no weather": ["temperature", "precipitation", "wind_speed", "is_raining"],
        "no holiday / day-off flags": ["is_holiday", "is_day_off"],
        "no hour-of-week profile": ["how_profile"],
    }
    for name, drop in variants.items():
        data = prepare_data(df, verbose=False)
        keep_h = [i for i, c in enumerate(data.hist_cols) if c not in drop]
        keep_f = [i for i, c in enumerate(data.fut_cols) if c not in drop]
        data.hist, data.hist_cols = data.hist[..., keep_h], [data.hist_cols[i] for i in keep_h]
        data.fut, data.fut_cols = data.fut[..., keep_f], [data.fut_cols[i] for i in keep_f]
        scores = []
        for seed in args.seeds:
            model, _, _ = train_member("gru_nograph", data, {}, seed, args, device)
            o = data.splits["test"]
            scores.append(compute_metrics(data.window(o, "y_raw"),
                                          neural_predict(model, data, o, None, device))["WMAPE"])
        rows.append({"variant": name, "test_WMAPE": fmt(np.mean(scores), np.std(scores)),
                     "wmape_mean": np.mean(scores)})
        print(f"  ablation {name}: {rows[-1]['test_WMAPE']}")
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if args.quick:
        args.seeds, args.epochs = args.seeds[:1], 3
    for d in (config.ARTIFACT_DIR, config.RESULTS_DIR, config.PLOT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    df = pd.read_parquet(config.DATASET_PATH)
    data = prepare_data(df)
    graphs = build_graphs(data)
    for name, (ei, _) in graphs.items():
        print(f"Graph '{name}': {ei.size(1)} directed edges")
    val_o, test_o = data.splits["val"], data.splits["test"]
    y_val, y_test = data.window(val_o, "y_raw"), data.window(test_o, "y_raw")

    per_seed = {}     # member -> list of test metric dicts (one per seed)
    val_preds, test_preds = {}, {}
    histories = {}

    # ── Baselines ────────────────────────────────────────────────────────────
    for name, fn in [("naive_24h", lambda o: seasonal_naive(data, o, 24)),
                     ("naive_168h", lambda o: seasonal_naive(data, o, 168)),
                     ("how_mean", lambda o: hour_of_week_mean(data, o))]:
        val_preds[name], test_preds[name] = fn(val_o), fn(test_o)
        per_seed[name] = [compute_metrics(y_test, test_preds[name])]

    xgb_runs = []
    for seed in args.seeds:
        m = XGBoostDirect(data, seed=seed).fit()
        xgb_runs.append(m)
        per_seed.setdefault("xgb_direct", []).append(compute_metrics(y_test, m.predict(test_o)))
    val_preds["xgb_direct"] = np.mean([m.predict(val_o) for m in xgb_runs], axis=0)
    test_preds["xgb_direct"] = np.mean([m.predict(test_o) for m in xgb_runs], axis=0)
    xgb_runs[0].model.save_model(config.ARTIFACT_DIR / "xgb_direct.json")
    print(f"XGBoost direct: test WMAPE {per_seed['xgb_direct'][0]['WMAPE']:.2f}%")

    # ── Neural models ────────────────────────────────────────────────────────
    for member in args.models:
        print(f"\nTraining {DISPLAY[member]}")
        vp, tp = [], []
        for seed in args.seeds:
            model, graph, hist = train_member(member, data, graphs, seed, args, device)
            histories[f"{member}_seed{seed}"] = hist
            torch.save(model.state_dict(), config.ARTIFACT_DIR / f"{member}_seed{seed}.pt")
            vp.append(neural_predict(model, data, val_o, graph, device))
            tp.append(neural_predict(model, data, test_o, graph, device))
            per_seed.setdefault(member, []).append(compute_metrics(y_test, tp[-1]))
        val_preds[member], test_preds[member] = np.mean(vp, axis=0), np.mean(tp, axis=0)

    # ── Ensemble (weights and residual model use validation data only) ───────
    members = [m for m in args.models] + ["xgb_direct", "how_mean"]
    weights = fit_weights({m: val_preds[m] for m in members}, y_val)
    print("\nEnsemble weights (fit on validation): " +
          ", ".join(f"{m} {w:.2f}" for m, w in weights.items() if w > 0.005))
    ens_val = combine(val_preds, weights)
    ens_test = combine(test_preds, weights)
    per_seed["ensemble"] = [compute_metrics(y_test, ens_test)]

    resid = ResidualCorrector(data, members).fit(
        val_o, {m: val_preds[m] for m in members}, ens_val, y_val)
    print(f"Residual corrector: {resid.report}")
    ens_resid_test = resid.predict(test_o, {m: test_preds[m] for m in members}, ens_test)
    if resid.enabled:
        per_seed["ensemble_resid"] = [compute_metrics(y_test, ens_resid_test)]
        resid.model.save_model(config.ARTIFACT_DIR / "residual.json")
    final_test = ens_resid_test

    # ── Tables ───────────────────────────────────────────────────────────────
    def table(select):
        rows = []
        for m, runs in per_seed.items():
            if select is not None:
                runs = [compute_metrics(y_test[select], p[select]) for p in
                        ([test_preds[m]] if m in test_preds else
                         [ens_test] if m == "ensemble" else [ens_resid_test])]
            row = {"model": DISPLAY[m], "id": m, "seeds": len(runs)}
            for k in runs[0]:
                vals = [r[k] for r in runs]
                row[k] = fmt(np.mean(vals), np.std(vals))
                row[f"{k}_mean"] = float(np.mean(vals))
            rows.append(row)
        return pd.DataFrame(rows).sort_values("WMAPE_mean").reset_index(drop=True)

    metrics = table(None)
    midnight = data.times[test_o].hour == 0
    metrics_da = table(midnight)
    metrics.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    metrics_da.to_csv(config.RESULTS_DIR / "metrics_dayahead.csv", index=False)
    cols = ["model", "MAE", "RMSE", "WMAPE", [c for c in metrics.columns if c.startswith("MAPE(")][0]]
    print("\nTest month (all hourly origins, mean ± std over seeds):")
    print(metrics[cols].to_string(index=False))
    print(f"\nDay-ahead only (forecast issued at 00:00, {midnight.sum()} days):")
    print(metrics_da[cols].to_string(index=False))

    ablation = run_ablation(df, args, device) if args.ablation else None
    if ablation is not None:
        ablation.to_csv(config.RESULTS_DIR / "ablation.csv", index=False)

    # ── Artifacts for the API ────────────────────────────────────────────────
    np.savez(config.ARTIFACT_DIR / "graphs.npz",
             **{f"{k}_index": v[0].numpy() for k, v in graphs.items()},
             **{f"{k}_weight": v[1].numpy() for k, v in graphs.items()})
    meta = {
        "stops": data.stops,
        "hist_cols": data.hist_cols, "fut_cols": data.fut_cols,
        "lookback": data.lookback, "horizon": data.horizon,
        "val_start": config.VAL_START, "test_start": config.TEST_START,
        "scaler": data.scaler.state_dict(),
        "graphs": list(graphs),
        "neural_members": {m: args.seeds for m in args.models},
        "model_kwargs": {"hidden_dim": config.TRAIN["hidden_dim"], "dropout": config.TRAIN["dropout"]},
        "weights": {m: w for m, w in weights.items() if w > 0},
        "residual_enabled": bool(resid.enabled),
        "residual_members": members,
    }
    # Zero-weight members are not needed at serving time unless the residual
    # corrector uses their forecasts as features.
    needed = set(members) if resid.enabled else set(meta["weights"])
    meta["neural_members"] = {m: s for m, s in meta["neural_members"].items() if m in needed}
    (config.ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    np.savez_compressed(config.RESULTS_DIR / "test_predictions.npz",
                        origins=test_o, times=data.times[test_o].values.astype("datetime64[s]"),
                        stops=np.array(data.stops), y_true=y_test, y_pred=final_test,
                        **{f"member_{m}": test_preds[m] for m in members})
    summary = {
        "test_period": [str(data.times[test_o[0]]), str(data.times[test_o[-1] + data.horizon - 1])],
        "n_stops": data.n_stops, "stops": data.stops,
        "samples": {k: int(len(v)) for k, v in data.splits.items()},
        "seeds": args.seeds, "weights": weights, "residual": resid.report,
        "metrics": metrics.to_dict("records"), "metrics_dayahead": metrics_da.to_dict("records"),
        "ablation": ablation.to_dict("records") if ablation is not None else None,
    }
    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # ── Plots ────────────────────────────────────────────────────────────────
    ev.plot_model_comparison(metrics)
    ev.plot_error_by_hour(y_test, final_test, data.times[test_o], "Ensemble")
    ev.plot_error_by_stop(y_test, final_test, data.stops, "Ensemble")
    ev.plot_forecast_day(y_test, final_test, data.times[test_o], data.stops, "Ensemble")
    for key, hist in histories.items():
        if key.endswith(f"seed{args.seeds[0]}"):
            ev.plot_training_history(hist, key)
    print(f"\nSaved results to {config.RESULTS_DIR} and artifacts to {config.ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
