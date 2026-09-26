import math

import numpy as np
import pandas as pd
import pytest

from orb_lab.engine.backtest import run_backtest
from orb_lab.engine.metrics import drawdown_stats, return_stats, trade_stats
from orb_lab.engine.params import SizingParams, StrategyParams

from .conftest import RANGE_BARS, day_bars, prepare

WIN = {**RANGE_BARS, "09:54": (100.0, 101.5, 100.0, 101.5), "10:10": (101.5, 104.25, 101.5, 104.0)}
LOSS = {**RANGE_BARS, "09:54": (100.0, 101.5, 100.0, 101.5), "10:10": (101.5, 101.5, 98.75, 99.0)}
FLAT = {}


def test_known_three_trade_sequence():
    days = [("2024-01-02", WIN), ("2024-01-03", LOSS), ("2024-01-04", FLAT), ("2024-01-05", WIN), ("2024-01-08", WIN)]
    prep = prepare([day_bars(d, spec) for d, spec in days])
    res = run_backtest(prep, StrategyParams(), sizing=SizingParams(starting_equity=10_000))
    m = res.metrics
    assert m["n_trades"] == 4
    assert m["win_rate"] == pytest.approx(0.75) and m["loss_rate"] == pytest.approx(0.25)
    assert m["profit_factor"] == pytest.approx(3.0)  # 375 / 125
    assert m["expectancy"] == pytest.approx(62.5)
    assert m["avg_winner"] == pytest.approx(125) and m["avg_loser"] == pytest.approx(-125)
    assert m["payoff_ratio"] == pytest.approx(1.0)
    assert m["avg_r"] == pytest.approx(0.5) and m["median_r"] == pytest.approx(1.0)
    assert m["longest_win_streak"] == 2 and m["longest_loss_streak"] == 1
    assert m["best_trade"] == 125 and m["worst_trade"] == -125
    assert m["net_pnl"] == pytest.approx(250) and m["end_equity"] == pytest.approx(10_250)
    assert m["max_dd"] == pytest.approx(10_000 / 10_125 - 1)
    assert m["max_dd_duration_days"] == 2  # below peak on 01-03 and 01-04
    assert res.daily["n_trades"].sum() == 4 and len(res.daily) == 5  # flat day included
    assert res.metrics_long["n_trades"] == 4 and res.metrics_short["n_trades"] == 0


def test_gross_vs_net():
    from .conftest import make_instrument

    inst = make_instrument(commission_per_side=2.0, slippage_ticks=1.0)
    prep = prepare([day_bars("2024-01-02", WIN)], instrument=inst)
    res = run_backtest(prep, StrategyParams())
    assert res.metrics_gross["net_pnl"] - res.metrics["net_pnl"] == pytest.approx(res.metrics["commission_paid"] + res.metrics["slippage_cost"])


def test_return_stats_math():
    pnl = np.array([100.0, -50.0, 0.0, 200.0])
    dates = pd.bdate_range("2024-01-01", periods=4)
    s = return_stats(pnl, dates, 1000.0)
    eq = 1000 + np.cumsum(pnl)
    rets = pnl / np.concatenate(([1000.0], eq[:-1]))
    assert s["sharpe"] == pytest.approx(rets.mean() / rets.std(ddof=1) * math.sqrt(252))
    assert s["total_return"] == pytest.approx(0.25)
    assert s["max_dd"] == pytest.approx(1050 / 1100 - 1)


def test_drawdown_duration():
    dd, dur = drawdown_stats(np.array([110.0, 100, 105, 111, 90, 95]), 100.0)
    assert dd == pytest.approx(90 / 111 - 1) and dur == 2


def test_trade_stats_empty():
    s = trade_stats(np.zeros(0), np.zeros(0), np.zeros(0), 1.0)
    assert s["n_trades"] == 0 and s["profit_factor"] == 0.0
