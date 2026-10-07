import numpy as np
import pytest
import torch

from models.stgnn import build_model
from utils.metrics import compute_metrics

B, N, L, H, FH, FF = 2, 5, 12, 6, 7, 4


def _inputs():
    torch.manual_seed(0)
    ei = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    return torch.randn(B, N, L, FH), torch.randn(B, N, H, FF), ei, torch.rand(ei.size(1))


@pytest.mark.parametrize("arch,extra", [("STGNN", {}), ("STGNN", {"use_graph": False}),
                                        ("STGNN_v3", {}), ("AGCRN", {})])
def test_forward_shapes(arch, extra):
    x_hist, x_fut, ei, ew = _inputs()
    m = build_model(arch, n_feats=FH, fut_feats=FF, n_stops=N, horizon=H, hidden_dim=16, **extra)
    assert m(x_hist, x_fut, ei, ew).shape == (B, N, H)


def test_gat_uses_edge_weights():
    x_hist, x_fut, ei, ew = _inputs()
    m = build_model("STGNN", n_feats=FH, fut_feats=FF, n_stops=N, horizon=H, hidden_dim=16).eval()
    with torch.no_grad():
        assert not torch.allclose(m(x_hist, x_fut, ei, ew), m(x_hist, x_fut, ei, ew * 0 + 5))


def test_metrics():
    y = np.array([0.0, 10.0, 20.0])
    p = np.array([2.0, 12.0, 16.0])
    m = compute_metrics(y, p)
    assert m["MAE"] == pytest.approx(8 / 3)
    assert m["WMAPE"] == pytest.approx(8 / 30 * 100)
    assert m["MAPE(y>=10)"] == pytest.approx((0.2 + 0.2) / 2 * 100)
    assert compute_metrics(y, -p)["MAE"] == pytest.approx(10.0)  # negatives clamped to 0
