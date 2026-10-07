"""
models/forecaster.py
--------------------
Model specs, graph construction and the saved-artifact format, shared by the
experiment runner (which trains and saves) and the API (which loads and serves).

Artifact directory layout (outputs/artifacts/):
  meta.json                   data setup, scaler, member list, ensemble weights
  graphs.npz                  edge_index / edge_weight per graph name
  <member>_seed<k>.pt         neural model state dicts (weights only)
  xgb_direct.json             XGBoostDirect booster
  residual.json               residual corrector booster (if enabled)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

import config
from models.baselines import XGBoostDirect, hour_of_week_mean
from models.ensemble import ResidualCorrector, combine
from models.stgnn import build_model
from utils.dataset import Log1pScaler, PreparedData, WindowDataset
from utils.graph_builder import od_graph, route_graph

# Neural members: id → (display name, architecture, graph, extra kwargs)
NEURAL_SPECS = {
    "gru_nograph": ("GRU (no graph)",           "STGNN",    None,    {"use_graph": False}),
    "stgnn_route": ("STGNN (route graph)",      "STGNN",    "route", {}),
    "stgnn_od":    ("STGNN (OD-flow graph)",    "STGNN",    "od",    {}),
    "stgnn_v3":    ("STGNN-v3 (BiLSTM+attn)",   "STGNN_v3", "route", {}),
    "agcrn":       ("AGCRN (learned graph)",    "AGCRN",    None,    {}),
}
DISPLAY = {k: v[0] for k, v in NEURAL_SPECS.items()}
DISPLAY.update({
    "naive_24h": "Seasonal naive (yesterday)",
    "naive_168h": "Seasonal naive (last week)",
    "how_mean": "Historical hour-of-week mean",
    "xgb_direct": "XGBoost (direct)",
    "ensemble": "Ensemble (weighted)",
    "ensemble_resid": "Ensemble + residual XGBoost",
})


def train_od_trips(data: PreparedData) -> pd.DataFrame:
    trips = pd.read_parquet(config.OD_TRIPS_PATH)
    train_end = data.times[data.splits["train"][-1] + data.horizon - 1]
    return trips[pd.to_datetime(trips["slot"]) <= train_end]


def build_graphs(data: PreparedData) -> dict:
    graphs = {"route": route_graph(data.stops, config.ROUTE_STOP_ORDER, config.ROUTE_IS_LOOP)}
    if config.OD_TRIPS_PATH.exists():
        graphs["od"] = od_graph(data.stops, train_od_trips(data))
    return graphs


def make_model(member: str, data: PreparedData, hidden_dim: int = config.TRAIN["hidden_dim"],
               dropout: float = config.TRAIN["dropout"]):
    _, arch, _, extra = NEURAL_SPECS[member]
    base = data.fut_cols.index("how_profile") if "how_profile" in data.fut_cols else None
    return build_model(arch, n_feats=len(data.hist_cols), fut_feats=len(data.fut_cols),
                       n_stops=data.n_stops, horizon=data.horizon,
                       hidden_dim=hidden_dim, dropout=dropout, base_index=base, **extra)


@torch.no_grad()
def neural_predict(model, data: PreparedData, origins, graph, device, batch_size=64):
    model.eval().to(device)
    ds = WindowDataset(data, origins)
    ei, ew = (graph[0].to(device), graph[1].to(device)) if graph is not None else (None, None)
    out = []
    for i in range(0, len(ds), batch_size):
        out.append(model(ds.x_hist[i:i + batch_size].to(device),
                         ds.x_fut[i:i + batch_size].to(device), ei, ew).cpu().numpy())
    return np.maximum(data.scaler.inverse(np.concatenate(out)), 0)


class Forecaster:
    """Loads saved artifacts and produces ensemble forecasts for any origins."""

    def __init__(self, data: PreparedData, artifact_dir: Path = config.ARTIFACT_DIR, device=None):
        artifact_dir = Path(artifact_dir)
        meta_path = artifact_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found. Train the models first: python scripts/run_experiments.py")
        self.meta = json.loads(meta_path.read_text())
        if self.meta["stops"] != data.stops:
            raise ValueError("Stops in the dataset differ from the trained models — retrain.")
        self.data = data
        data.scaler = Log1pScaler.from_state_dict(self.meta["scaler"])
        self.device = device or torch.device("cpu")
        g = np.load(artifact_dir / "graphs.npz")
        self.graphs = {k: (torch.from_numpy(g[f"{k}_index"]), torch.from_numpy(g[f"{k}_weight"]))
                       for k in self.meta["graphs"]}

        self.neural = {}
        for member, seeds in self.meta["neural_members"].items():
            models = []
            for seed in seeds:
                path = artifact_dir / f"{member}_seed{seed}.pt"
                if not path.exists():
                    raise FileNotFoundError(f"Missing checkpoint {path}")
                m = make_model(member, data, **self.meta["model_kwargs"])
                m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                models.append(m)
            self.neural[member] = models

        self.xgb_direct = None
        if "xgb_direct" in self.meta["weights"]:
            self.xgb_direct = XGBoostDirect(data)
            self.xgb_direct.model.load_model(artifact_dir / "xgb_direct.json")

        self.residual = None
        if self.meta.get("residual_enabled"):
            self.residual = ResidualCorrector(data, self.meta["residual_members"])
            self.residual.model.load_model(artifact_dir / "residual.json")
            self.residual.enabled = True

    def member_predictions(self, origins: np.ndarray) -> dict:
        preds = {}
        for member, models in self.neural.items():
            graph_name = NEURAL_SPECS[member][2]
            graph = self.graphs.get(graph_name) if graph_name else None
            preds[member] = np.mean(
                [neural_predict(m, self.data, origins, graph, self.device) for m in models], axis=0)
        if self.xgb_direct is not None:
            preds["xgb_direct"] = self.xgb_direct.predict(origins)
        preds["how_mean"] = hour_of_week_mean(self.data, origins)
        return preds

    def predict(self, origins: np.ndarray) -> np.ndarray:
        origins = np.asarray(origins)
        preds = self.member_predictions(origins)
        ens = combine(preds, self.meta["weights"])
        if self.residual is not None:
            ens = self.residual.predict(origins, preds, ens)
        return ens
