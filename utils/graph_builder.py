"""
utils/graph_builder.py
----------------------
Stop graphs for the GNN models. Each builder returns (edge_index, edge_weight)
for a list of stops; node i is stops[i].

  route_graph     consecutive stops along the physical route (loop-aware)
  od_graph        stops linked by passenger flow (top-k destinations per stop)
  distance_graph  Gaussian-kernel graph from the UrbanBus distance matrix
"""

import numpy as np
import pandas as pd
import torch

from utils.route import od_count_matrix


def _to_tensors(src, dst, w):
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    if edge_weight.numel():
        edge_weight = edge_weight / edge_weight.max()
    return edge_index, edge_weight


def route_graph(stops: list, stop_order: list, is_loop: bool):
    """Bidirectional edges between stops that are consecutive along the route."""
    ordered = [s for s in stop_order if s in stops]
    idx = {s: i for i, s in enumerate(stops)}
    pairs = list(zip(ordered[:-1], ordered[1:]))
    if is_loop and len(ordered) > 2:
        pairs.append((ordered[-1], ordered[0]))
    src, dst = [], []
    for a, b in pairs:
        src += [idx[a], idx[b]]
        dst += [idx[b], idx[a]]
    return _to_tensors(src, dst, [1.0] * len(src))


def od_graph(stops: list, trips: pd.DataFrame, k: int = 3):
    """
    Links each stop to the k stops it exchanges the most passengers with
    (flow in both directions). Weight = share of the stop's total exchange.
    Use training-period trips only.
    """
    c = od_count_matrix(trips, stops)
    np.fill_diagonal(c, 0)
    flow = c + c.T
    share = flow / np.maximum(flow.sum(axis=1, keepdims=True), 1)
    edges = {}
    for i in range(len(stops)):
        for j in np.argsort(-share[i])[:k]:
            if share[i, j] <= 0:
                continue
            for a, b in ((i, int(j)), (int(j), i)):
                edges[(a, b)] = max(edges.get((a, b), 0), share[i, j])
    src, dst = zip(*edges) if edges else ((), ())
    return _to_tensors(list(src), list(dst), list(edges.values()))


def distance_graph(stops: list, dist_matrix_path, sigma_km: float = None, threshold: float = 0.3):
    """
    Gaussian kernel w_ij = exp(-(d_ij / sigma)^2), edges kept where w ≥ threshold.
    sigma defaults to the std of the pairwise distances (Li et al., DCRNN).
    Requires data/raw/DistanceMatrix.csv from the UrbanBus dataset.
    """
    dm = pd.read_csv(dist_matrix_path, index_col=0)
    missing = [s for s in stops if s not in dm.index]
    if missing:
        raise KeyError(f"Stops missing from distance matrix: {missing}")
    d = dm.loc[stops, stops].values.astype(float)
    d = np.minimum(d, d.T)                           # matrix may be asymmetric
    sigma = sigma_km or d[np.triu_indices_from(d, 1)].std()
    w = np.exp(-(d / sigma) ** 2)
    np.fill_diagonal(w, 0)
    src, dst = np.nonzero(w >= threshold)
    return _to_tensors(src.tolist(), dst.tolist(), w[src, dst].tolist())
