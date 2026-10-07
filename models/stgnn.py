"""
models/stgnn.py
---------------
Spatiotemporal models for day-ahead (24 h) boarding forecasts per stop.

All models share one interface:

    y = model(x_hist, x_fut, edge_index, edge_weight)
        x_hist [B, N, L, F_hist]   observed history per stop
        x_fut  [B, N, H, F_fut]    known-at-forecast-time features per target hour
        y      [B, N, H]           scaled boardings

Models
  STGNN     GRU temporal encoder → 2 × GAT (edge-weighted) → decoder.
            use_graph=False gives the no-graph ablation (same network minus GAT).
  STGNN_v3  BiLSTM + multi-scale attention → GAT → gated fusion → decoder.
  AGCRN     graph-convolutional GRU with a learned adjacency and node-adaptive
            parameters (Bai et al., NeurIPS 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


def batch_graph(edge_index, edge_weight, B: int, N: int):
    """B disjoint copies of the graph: node ids offset by b*N."""
    E = edge_index.size(1)
    offsets = (torch.arange(B, device=edge_index.device) * N).repeat_interleave(E)
    ei = edge_index.repeat(1, B) + offsets
    ew = edge_weight.repeat(B) if edge_weight is not None else None
    return ei, ew


class ForecastDecoder(nn.Module):
    """
    Per-horizon decoder that combines the stop's encoded state with what is
    known about each target hour (calendar, same hour yesterday / last week,
    hour-of-week profile).

      z_h = W_h · state  +  W_f · x_fut[h]           (horizon-specific)
      y_h = MLP(z_h) + linear(x_fut[h])              (linear skip keeps the
                                                      seasonal-naive path easy)
    """

    def __init__(self, hidden_dim: int, fut_dim: int, horizon: int, dropout: float = 0.1,
                 base_index: int = None):
        super().__init__()
        self.horizon, self.d = horizon, hidden_dim
        self.state_proj = nn.Linear(hidden_dim, horizon * hidden_dim)
        self.fut_proj = nn.Linear(fut_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.skip = nn.Linear(fut_dim, 1)
        if base_index is not None:
            # Start as the historical hour-of-week profile and learn corrections:
            # an untrained model already equals that strong baseline.
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)
            with torch.no_grad():
                self.skip.weight[0, base_index] = 1.0
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, x_fut):
        B, N, _ = h.shape
        z = self.state_proj(h).view(B, N, self.horizon, self.d) + self.fut_proj(x_fut)
        return (self.mlp(z) + self.skip(x_fut)).squeeze(-1)


class TemporalEncoder(nn.Module):
    """GRU over each stop's history. [B, N, L, F] → [B, N, hidden]."""

    def __init__(self, n_feats: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_feats, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, N, T, F_ = x.shape
        out, _ = self.gru(x.reshape(B * N, T, F_))
        return self.norm(out[:, -1]).view(B, N, -1)


class SpatialEncoder(nn.Module):
    """Two residual GAT layers; edge weights enter the attention as edge features."""

    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout,
                            concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout,
                            concat=False, edge_dim=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_weight):
        attr = edge_weight.view(-1, 1) if edge_weight is not None else None
        h = self.norm1(h + F.elu(self.gat1(h, edge_index, edge_attr=attr)))
        h = self.norm2(h + F.elu(self.gat2(h, edge_index, edge_attr=attr)))
        return h


class STGNN(nn.Module):
    def __init__(self, n_feats: int, fut_feats: int, n_stops: int, horizon: int,
                 hidden_dim: int = 64, gru_layers: int = 2, gat_heads: int = 4,
                 dropout: float = 0.1, use_graph: bool = True, base_index: int = None):
        super().__init__()
        self.use_graph = use_graph
        self.temporal = TemporalEncoder(n_feats, hidden_dim, gru_layers, dropout)
        self.spatial = SpatialEncoder(hidden_dim, gat_heads, dropout) if use_graph else None
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.decoder = ForecastDecoder(hidden_dim, fut_feats, horizon, dropout, base_index)

    def forward(self, x_hist, x_fut, edge_index=None, edge_weight=None):
        B, N = x_hist.shape[:2]
        h = self.temporal(x_hist)
        if self.use_graph:
            ei, ew = batch_graph(edge_index, edge_weight, B, N)
            h_spat = self.spatial(h.reshape(B * N, -1), ei, ew).view(B, N, -1)
            h = self.fusion_norm(h + h_spat)
        return self.decoder(h, x_fut)


class MultiScaleAttention(nn.Module):
    """
    Attention from the last hidden state over the last 6 h, 24 h and the full
    lookback, each scale with its own attention module, mixed by learned weights.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, windows=(6, 24, None)):
        super().__init__()
        self.windows = windows
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True) for _ in windows)
        self.scale_logits = nn.Parameter(torch.zeros(len(windows)))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_seq):
        T = h_seq.shape[1]
        query = h_seq[:, -1:]
        outs = []
        for attn, w in zip(self.attn, self.windows):
            kv = h_seq[:, max(0, T - w):] if w else h_seq
            outs.append(attn(query, kv, kv, need_weights=False)[0].squeeze(1))
        weights = torch.softmax(self.scale_logits, dim=0)
        return self.norm(sum(w * o for w, o in zip(weights, outs)))


class BiLSTMEncoder(nn.Module):
    def __init__(self, n_feats: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_feats, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.bi_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attention = MultiScaleAttention(hidden_dim)

    def forward(self, x):
        B, N, T, F_ = x.shape
        h, _ = self.lstm(self.input_proj(x.reshape(B * N, T, F_)))
        return self.attention(self.bi_proj(h)).view(B, N, -1)


class STGNN_v3(nn.Module):
    """BiLSTM + multi-scale attention → GAT → gated temporal/spatial fusion."""

    def __init__(self, n_feats: int, fut_feats: int, n_stops: int, horizon: int,
                 hidden_dim: int = 64, lstm_layers: int = 2, gat_heads: int = 4,
                 dropout: float = 0.1, base_index: int = None):
        super().__init__()
        self.temporal = BiLSTMEncoder(n_feats, hidden_dim, lstm_layers, dropout)
        self.spatial = SpatialEncoder(hidden_dim, gat_heads, dropout)
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.decoder = ForecastDecoder(hidden_dim, fut_feats, horizon, dropout, base_index)

    def forward(self, x_hist, x_fut, edge_index=None, edge_weight=None):
        B, N = x_hist.shape[:2]
        h_temp = self.temporal(x_hist)
        ei, ew = batch_graph(edge_index, edge_weight, B, N)
        h_spat = self.spatial(h_temp.reshape(B * N, -1), ei, ew).view(B, N, -1)
        g = self.gate(torch.cat([h_temp, h_spat], dim=-1))
        return self.decoder(self.fusion_norm(g * h_temp + (1 - g) * h_spat), x_fut)


# ── AGCRN ─────────────────────────────────────────────────────────────────────
class AVWGCN(nn.Module):
    """
    Adaptive graph convolution with node-adaptive weights:
      A = softmax(ReLU(E Eᵀ)),  supports = [I, A, 2A·A − I, ...] (Chebyshev)
      W_n = E_n · W_pool,  b_n = E_n · b_pool
    The supports and weights depend only on E, so they are computed once per
    forward pass (prepare) and reused at every time step.
    """

    def __init__(self, in_dim: int, out_dim: int, embed_dim: int, cheb_k: int = 2):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, cheb_k, in_dim, out_dim))
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, out_dim))
        nn.init.xavier_uniform_(self.weights_pool.view(embed_dim * cheb_k, in_dim, out_dim))

    def prepare(self, E):
        N = E.shape[0]
        A = F.softmax(F.relu(E @ E.T), dim=1)
        supports = [torch.eye(N, device=E.device), A]
        for _ in range(2, self.cheb_k):
            supports.append(2 * A @ supports[-1] - supports[-2])
        S = torch.stack(supports[:self.cheb_k])                      # [K, N, N]
        W = torch.einsum("nd,dkio->nkio", E, self.weights_pool)      # [N, K, C, O]
        return S, W, E @ self.bias_pool                              # b [N, O]

    def forward(self, x, S, W, b):                                   # x [B, N, C]
        x_g = torch.einsum("knm,bmc->bnkc", S, x)                    # [B, N, K, C]
        return torch.einsum("bnkc,nkco->bno", x_g, W) + b


class AGCRNCell(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embed_dim: int, cheb_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate = AVWGCN(in_dim + hidden_dim, 2 * hidden_dim, embed_dim, cheb_k)
        self.update = AVWGCN(in_dim + hidden_dim, hidden_dim, embed_dim, cheb_k)

    def prepare(self, E):
        return self.gate.prepare(E), self.update.prepare(E)

    def forward(self, x, h, params):
        (gS, gW, gb), (uS, uW, ub) = params
        zr = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1), gS, gW, gb))
        z, r = zr.split(self.hidden_dim, dim=-1)
        h_tilde = torch.tanh(self.update(torch.cat([x, r * h], dim=-1), uS, uW, ub))
        return z * h + (1 - z) * h_tilde


class AGCRN(nn.Module):
    """Graph is learned from data; edge_index / edge_weight are ignored."""

    def __init__(self, n_feats: int, fut_feats: int, n_stops: int, horizon: int,
                 hidden_dim: int = 64, embed_dim: int = 10, n_layers: int = 2,
                 cheb_k: int = 2, dropout: float = 0.1, base_index: int = None):
        super().__init__()
        self.node_embed = nn.Parameter(torch.randn(n_stops, embed_dim))
        dims = [n_feats] + [hidden_dim] * n_layers
        self.cells = nn.ModuleList(
            AGCRNCell(dims[i], hidden_dim, embed_dim, cheb_k) for i in range(n_layers))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.decoder = ForecastDecoder(hidden_dim, fut_feats, horizon, dropout, base_index)
        self.hidden_dim = hidden_dim

    def forward(self, x_hist, x_fut, edge_index=None, edge_weight=None):
        B, N, T, _ = x_hist.shape
        seq = x_hist
        for cell in self.cells:
            params = cell.prepare(self.node_embed)
            h = x_hist.new_zeros(B, N, self.hidden_dim)
            outs = []
            for t in range(T):
                h = cell(seq[:, :, t], h, params)
                outs.append(h)
            seq = self.dropout(torch.stack(outs, dim=2))
        return self.decoder(self.norm(seq[:, :, -1]), x_fut)


MODEL_REGISTRY = {"STGNN": STGNN, "STGNN_v3": STGNN_v3, "AGCRN": AGCRN}


def build_model(arch: str, **kwargs) -> nn.Module:
    return MODEL_REGISTRY[arch](**kwargs)
