import numpy as np
import pandas as pd
import pytest

from optimize.frequency import FrequencyOptimizer
from utils.route import ODLoadModel, infer_stop_order, segment_incidence


def test_incidence_linear_and_loop():
    lin = segment_incidence(["a", "b", "c", "d"], is_loop=False)
    assert lin.shape == (4, 4, 3)
    assert lin[0, 2].tolist() == [1, 1, 0]
    assert lin[2, 0].sum() == 0                      # backward trip impossible on a line
    loop = segment_incidence(["a", "b", "c", "d"], is_loop=True)
    assert loop.shape == (4, 4, 4)
    assert loop[2, 0].tolist() == [0, 0, 1, 1]       # c→d→a wraps around


def _trips(order, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    o = rng.integers(0, len(order) - 1, n)
    d = o + 1 + (rng.integers(0, 10**6, n) % (len(order) - 1 - o))
    return pd.DataFrame({"boarding_stop": [order[i] for i in o], "alighting_stop": [order[j] for j in d],
                         "slot": pd.Timestamp("2020-01-01 08:00")})


def test_infer_stop_order_recovers_route():
    order = ["Q", "C", "X", "A", "M", "B"]
    found, fwd = infer_stop_order(_trips(order))
    assert found == order and fwd == 1.0


def test_load_model_conserves_passengers():
    order = ["a", "b", "c", "d"]
    m = ODLoadModel(order, is_loop=True, min_trips=1).fit(_trips(order))
    board = np.array([[10.0], [5.0], [3.0], [0.0]])
    assert np.isclose(m.alighting(board, [8]).sum(), board.sum())
    load = m.segment_load(board, [8])
    # everyone boarding at 'a' crosses segment a→b
    assert load[0, 0] >= 10 - 1e-6


def test_optimizer_meets_capacity():
    opt = FrequencyOptimizer(bus_capacity=80, min_headway=5, max_headway=30, round_trip_min=60)
    load = np.array([[0, 100, 500, 50], [10, 300, 200, 20]], dtype=float)
    r = opt.optimize(load, load.sum(axis=0), fixed_headway=15)
    peak = load.max(axis=0)
    assert (r["optimized"]["departures"] * 80 >= peak).all()
    assert r["optimized"]["overcrowded_hours"] == 0
    assert r["fixed"]["overcrowded_hours"] == 1                     # 500 > 4 × 80
    assert (r["optimized"]["headway"] <= 30).all() and (r["optimized"]["headway"] >= 5).all()
    assert r["optimized"]["vehicles"][0] == 2                      # 30-min headway, 60-min loop


def test_optimizer_flags_capacity_limited_hours():
    opt = FrequencyOptimizer(bus_capacity=50, min_headway=10)
    r = opt.optimize(np.array([[1000.0]]), np.array([1000.0]))
    assert r["optimized"]["capacity_limited_hours"] == 1
    assert r["optimized"]["overcrowded_hours"] == 1


@pytest.mark.parametrize("kwargs", [dict(bus_capacity=0), dict(min_headway=40, max_headway=30),
                                    dict(round_trip_min=0)])
def test_optimizer_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        FrequencyOptimizer(**kwargs)


def test_fixed_headway_zero_rejected():
    with pytest.raises(ValueError):
        FrequencyOptimizer().optimize(np.ones((2, 3)), np.ones(3), fixed_headway=0)


def test_realized_evaluation_uses_actual_demand():
    opt = FrequencyOptimizer(bus_capacity=80)
    forecast = np.array([[100.0, 100.0]])
    actual = np.array([[100.0, 400.0]])          # second hour much busier than forecast
    r = opt.optimize(forecast, forecast[0], fixed_headway=15, actual_load=actual, actual_boardings=actual[0])
    assert r["optimized"]["overcrowded_hours"] == 0              # against the forecast
    assert r["optimized"]["realized_overcrowded_hours"] == 1     # against what happened
