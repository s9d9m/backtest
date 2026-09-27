"""Cost units and risk-based sizing: $ per unit per side, contracts vs shares, never mixed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orb_lab.engine.backtest import run_backtest
from orb_lab.engine.execution import resolve_costs
from orb_lab.engine.instruments import Instrument, load_instruments
from orb_lab.engine.params import ExecutionParams, ParamError, SizingParams, StrategyParams
from orb_lab.engine.portfolio import apply_sizing
from orb_lab.phase0 import pipeline as p0

from .conftest import day_bars, make_instrument, prepare

DATE = "2024-03-05"
LONG_SIGNAL = {"09:30": (100.0, 100.5, 99.5, 100.25), "09:44": (100.25, 101.0, 100.0, 100.75),
               "10:04": (100.75, 101.75, 100.75, 101.5)}


def _one_trade(inst, sizing=None, execution=None):
    prep = prepare(day_bars(DATE, {**LONG_SIGNAL, "10:10": (101.5, 104.75, 101.5, 104.5)}), inst)
    res = run_backtest(prep, StrategyParams(range_minutes=15, entry_tf=5, target_r=1.0), execution or ExecutionParams(),
                       sizing or SizingParams())
    return res, res.trades.iloc[0]


def test_every_cost_category_is_per_unit_per_side_and_scales_with_quantity():
    inst = make_instrument(commission_per_side=1.0, exchange_fee_per_side=0.5, friction_per_side=0.25, slippage_ticks=1.0)
    _, t1 = _one_trade(inst, SizingParams(contracts=1))
    _, t3 = _one_trade(inst, SizingParams(contracts=3))
    assert t1.commission == pytest.approx(2.0) and t1.fees == pytest.approx(1.0) and t1.friction == pytest.approx(0.5)
    assert t1.slippage_cost == pytest.approx(12.5)  # 1 tick on the market entry only (target is a limit)
    assert t1.total_cost == pytest.approx(2.0 + 1.0 + 0.5 + 12.5)
    for col in ("commission", "fees", "friction", "slippage_cost", "total_cost", "gross_pnl", "net_pnl", "risk_dollars"):
        assert t3[col] == pytest.approx(3 * t1[col]), col
    assert t1.gross_pnl - t1.total_cost == pytest.approx(t1.net_pnl)


def test_etf_cost_of_two_cents_per_share_is_never_per_order():
    cfg = p0.load_config()
    spy = p0.etf_instrument("SPY", cfg, 0.02)
    assert spy.unit == "share" and spy.asset_class == "etf"
    costs = resolve_costs(spy, ExecutionParams())
    assert costs.round_trip_fixed == pytest.approx(0.04)  # per share, both fills
    # 250 shares -> 250 x 2 x 0.02 = $10, not $0.04 (per order) and not $0.02 x 2 x 1
    trades = pd.DataFrame({"entry_price": [500.0], "risk_per_contract": [0.5], "net_pnl_pc": [0.46], "gross_pnl_pc": [0.5],
                           "slippage_pc": [0.0], "commission_pc": [0.0], "fees_pc": [0.0], "friction_pc": [0.04],
                           "fixed_cost_pc": [0.04], "total_cost_pc": [0.04]})
    out = apply_sizing(trades, SizingParams(contracts=250), spy, costs)
    assert out["friction"].iloc[0] == pytest.approx(10.0) and out["total_cost"].iloc[0] == pytest.approx(10.0)


def test_futures_and_etf_units_are_distinct():
    inst = load_instruments()
    assert inst["ES"].unit == "contract" and inst["ES"].asset_class == "future"
    assert inst["MES"].tick_value == pytest.approx(1.25) and inst["MES"].tick_size == inst["ES"].tick_size
    assert "per contract" not in inst["ES"].cost_description() or "/contract/side" in inst["ES"].cost_description()
    assert "/share/side" in p0.etf_instrument("SPY", p0.load_config(), 0.02).cost_description()
    with pytest.raises(ValueError):
        make_instrument(min_qty=0.5, qty_step=0.5)  # fractional futures contracts are not allowed
    with pytest.raises(ValueError):
        make_instrument(asset_class="stock")
    with pytest.raises(ValueError):
        make_instrument(commission_per_side=-1.0)


def test_tick_value_must_match_tick_size_times_multiplier():
    with pytest.raises(ValueError):
        make_instrument(tick_size=0.25, tick_value=5.0, multiplier=50)


def test_risk_pct_sizing_uses_stop_distance_and_floors_to_whole_units():
    inst = make_instrument()
    # risk per contract = 8 ticks x $12.50 = $100; 0.25 % of $40,000 = $100 -> 1; 1 % = $400 -> 4; 0.9 % = $360 -> floor(3.6) = 3
    for pct, qty in ((0.0025, 1), (0.01, 4), (0.009, 3)):
        res, t = _one_trade(inst, SizingParams(mode="pct_equity", risk_pct=pct, starting_equity=40_000))
        assert t.risk_per_contract == pytest.approx(100.0) and t.qty == qty and t.risk_dollars == pytest.approx(100.0 * qty)
    res, t = _one_trade(inst, SizingParams(mode="fixed_risk", risk_dollars=90.0, starting_equity=40_000))
    assert t.sized_out and t.qty == 0 and res.diagnostics["sized_out_trades"] == 1


def test_compounding_changes_size_with_equity():
    inst = make_instrument()
    trades = pd.DataFrame({"entry_price": [100.0] * 3, "risk_per_contract": [100.0] * 3, "net_pnl_pc": [100.0, 100.0, -100.0],
                           "gross_pnl_pc": [100.0, 100.0, -100.0], "slippage_pc": [0.0] * 3, "commission_pc": [0.0] * 3})
    costs = resolve_costs(inst, ExecutionParams())
    comp = apply_sizing(trades, SizingParams(mode="pct_equity", risk_pct=0.01, starting_equity=100_000), inst, costs)
    fixed = apply_sizing(trades, SizingParams(mode="fixed_risk", risk_dollars=1000, starting_equity=100_000), inst, costs)
    assert comp["qty"].tolist() == [10, 10, 10]  # 1 % of 101,000 = 1,010 -> still 10
    assert fixed["qty"].tolist() == [10, 10, 10]
    comp2 = apply_sizing(trades.assign(net_pnl_pc=[1000.0, 1000.0, -100.0]), SizingParams(mode="pct_equity", risk_pct=0.01,
                                                                                          starting_equity=100_000), inst, costs)
    assert comp2["qty"].tolist() == [10, 11, 12]  # equity 100k -> 110k (1,100 risk) -> 121k (1,210 risk)


def test_notional_leverage_cap_limits_etf_share_count():
    spy = p0.etf_instrument("SPY", p0.load_config(), 0.02)
    assert spy.max_notional_leverage == 4
    # $400 risk with a $0.05 stop would be 8,000 shares = $4M notional on $40k: capped at 4x -> 320 shares
    trades = pd.DataFrame({"entry_price": [500.0], "risk_per_contract": [0.05], "net_pnl_pc": [0.05], "gross_pnl_pc": [0.09],
                           "slippage_pc": [0.0], "commission_pc": [0.0], "friction_pc": [0.04], "fixed_cost_pc": [0.04],
                           "total_cost_pc": [0.04], "fees_pc": [0.0]})
    out = apply_sizing(trades, SizingParams(mode="pct_equity", risk_pct=0.01, starting_equity=40_000, max_contracts=1e9), spy,
                       resolve_costs(spy, ExecutionParams()))
    assert out["qty"].iloc[0] == 320 and out["size_capped"].iloc[0] and out["notional"].iloc[0] == pytest.approx(160_000)
    out = apply_sizing(trades, SizingParams(mode="pct_equity", risk_pct=0.01, starting_equity=40_000, max_contracts=1e9, max_leverage=0),
                       spy, resolve_costs(spy, ExecutionParams()))
    assert out["qty"].iloc[0] == 8000 and not out["size_capped"].iloc[0]


def test_frictionless_zeroes_every_category():
    inst = make_instrument(commission_per_side=1.0, exchange_fee_per_side=0.5, friction_per_side=0.25, slippage_ticks=1.0)
    _, t = _one_trade(inst, execution=ExecutionParams(frictionless=True))
    assert t.total_cost == 0 and t.commission == 0 and t.fees == 0 and t.friction == 0 and t.slippage_cost == 0


def test_invalid_sizing_and_costs_are_rejected():
    with pytest.raises(ParamError):
        SizingParams(mode="pct_equity", risk_pct=1.5).validate()
    with pytest.raises(ParamError):
        SizingParams(max_leverage=-1).validate()
    with pytest.raises(ParamError):
        ExecutionParams(friction_per_side=-0.01).validate()


def test_metrics_report_cost_breakdown_and_sides():
    inst = make_instrument(commission_per_side=1.0, exchange_fee_per_side=0.5, slippage_ticks=1.0)
    res, t = _one_trade(inst)
    m = res.metrics
    assert m["total_cost"] == pytest.approx(t.total_cost) and m["cost_per_trade"] == pytest.approx(t.total_cost)
    assert m["commission_paid"] + m["fees_paid"] + m["friction_paid"] + m["slippage_cost"] == pytest.approx(m["total_cost"])
    assert m["n_long"] == 1 and m["n_short"] == 0 and m["pnl_long"] == pytest.approx(t.net_pnl)
    assert res.metrics_gross["net_pnl"] - m["net_pnl"] == pytest.approx(m["total_cost"])
    assert np.isclose(m["cum_r"], t.r_multiple)
