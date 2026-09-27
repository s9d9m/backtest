"""Exact-outcome tests of the basic breakout kernel on hand-built sessions.

Standard setup: OR 09:30-09:45 ET with high 101.00 / low 99.00 (8 ticks of 0.25). A 5-minute entry bucket
[09:50, 09:55) closing at 101.50 is a long signal known at 09:55; the market order fills at the 09:55 open.
Stop at OR low 99.00 -> risk 2.50 points (10 ticks); a 1R target is 104.00.
"""

import numpy as np
import pandas as pd
import pytest

from orb_lab.engine.backtest import run_backtest
from orb_lab.engine.params import ExecutionParams, ParamError, SizingParams, StrategyParams

from .conftest import RANGE_BARS, day_bars, make_instrument, prepare

DATE = "2024-01-03"
LONG_SIGNAL = {**RANGE_BARS, "09:54": (100.0, 101.5, 100.0, 101.5)}
SHORT_SIGNAL = {**RANGE_BARS, "09:54": (100.0, 100.0, 98.5, 98.5)}
BASE = StrategyParams(orb_start="09:30", range_minutes=15, entry_tf=5, stop_method="or_opposite", target_r=1.0, cutoff="11:00")
EXEC = ExecutionParams(fill_model="conservative", ambiguity="conservative")


def run(bars, params=BASE, execution=EXEC, sizing=None, instrument=None, **prep_kwargs):
    prep = prepare(bars, instrument=instrument, **prep_kwargs)
    return run_backtest(prep, params, execution, sizing or SizingParams())


def only_trade(result):
    assert len(result.trades) == 1, result.trades
    return result.trades.iloc[0]


def test_opening_range_uses_only_window_bars():
    bars = day_bars(DATE, {"09:29": (100, 200, 1, 100), "09:30": (100, 101, 99, 100), "09:44": (100, 100.5, 99.5, 100), "09:45": (100, 150, 50, 100)})
    prep = prepare(bars)
    hi, lo, cov, ok = prep.opening_range(570, 15)
    assert ok[0] and cov[0] == 1.0
    assert hi[0] * 0.25 == 101.0 and lo[0] * 0.25 == 99.0
    hi5, lo5, _, _ = prep.opening_range(570 + 14, 1)  # just the 09:44 bar
    assert hi5[0] * 0.25 == 100.5 and lo5[0] * 0.25 == 99.5


def test_or_coverage_requirement():
    bars = day_bars(DATE, RANGE_BARS)
    bars = bars.drop(bars.index[31:36])  # remove 09:31..09:35 -> 10 of 15 bars left
    assert not prepare(bars, min_or_coverage=0.8).opening_range(570, 15)[3][0]
    assert prepare(bars, min_or_coverage=0.6).opening_range(570, 15)[3][0]


def test_long_breakout_market_entry_hits_target():
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:10": (101.5, 104.25, 101.5, 104.0)})
    t = only_trade(run(bars))
    assert t.direction == "long"
    assert t.signal_time.strftime("%H:%M") == "09:55" and t.entry_time.strftime("%H:%M") == "09:55"
    assert t.entry_price == 101.5 and t.stop_price == 99.0 and t.target_price == 104.0
    assert t.exit_reason == "target" and t.exit_price == 104.0
    assert t.net_pnl == pytest.approx(2.5 * 50)
    assert t.r_multiple == pytest.approx(1.0)


def test_short_breakout_market_entry():
    bars = day_bars(DATE, {**SHORT_SIGNAL, "10:10": (98.5, 98.5, 95.75, 96.0)})
    t = only_trade(run(bars))
    assert t.direction == "short" and t.entry_price == 98.5
    assert t.stop_price == 101.0 and t.target_price == 96.0
    assert t.exit_reason == "target" and t.net_pnl == pytest.approx(125.0)


def test_no_breakout_when_close_equals_boundary():
    bars = day_bars(DATE, {**RANGE_BARS, "09:54": (100.0, 101.25, 100.0, 101.0)})  # wicks above, closes AT high
    assert len(run(bars).trades) == 0


def test_bar_before_range_end_cannot_signal():
    # A spike inside the range window only widens the range; it is not a breakout.
    bars = day_bars(DATE, {"09:30": (100, 101, 99, 100), "09:40": (100, 103, 100, 102.5), "09:41": (102.5, 102.5, 100, 100)})
    assert len(run(bars).trades) == 0


def test_stop_loss_fill_and_gap_through_stop():
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:10": (101.5, 101.5, 98.75, 99.0)})
    t = only_trade(run(bars))
    assert t.exit_reason == "stop" and t.exit_price == 99.0 and t.net_pnl == pytest.approx(-125.0)
    gap = day_bars(DATE, {**LONG_SIGNAL, "10:10": (98.0, 98.0, 97.0, 97.5)})
    t = only_trade(run(gap))
    assert t.exit_reason == "stop" and t.exit_price == 98.0  # filled at the gap open, worse than the stop


def test_target_touch_standard_vs_conservative():
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:10": (101.5, 104.0, 101.5, 104.0)})
    std = only_trade(run(bars, execution=ExecutionParams(fill_model="standard", ambiguity="conservative")))
    assert std.exit_reason == "target" and std.exit_time.strftime("%H:%M") == "10:10"
    cons = only_trade(run(bars, execution=EXEC))
    # a touch is not a fill under the conservative model; price never trades through, so time exit at 15:55
    assert cons.exit_reason == "time" and cons.exit_time.strftime("%H:%M") == "15:55"


@pytest.mark.parametrize(
    "mode,reason,price",
    [("optimistic", "target", 104.0), ("pessimistic", "stop", 99.0), ("conservative", "stop", 99.0)],
)
def test_same_bar_stop_and_target_ambiguity(mode, reason, price):
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:10": (101.5, 104.5, 98.5, 100.0)})
    res = run(bars, execution=ExecutionParams(fill_model="conservative", ambiguity=mode))
    t = only_trade(res)
    assert t.exit_reason == reason and t.exit_price == price
    assert bool(t.ambiguous_exit)
    assert res.diagnostics["ambiguous_exits"] == 1


def test_open_decides_sequence_so_not_ambiguous():
    # Bar opens beyond the target: the target is reached first regardless of the low.
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:10": (104.5, 104.75, 98.0, 99.0)})
    t = only_trade(run(bars))
    assert t.exit_reason == "target" and t.exit_price == 104.0 and not t.ambiguous_exit


def test_entry_cutoff():
    late = day_bars(DATE, {**RANGE_BARS, "11:03": (100.0, 101.5, 100.0, 101.5)})  # bucket closes 11:05
    assert len(run(late).trades) == 0
    ok = day_bars(DATE, {**RANGE_BARS, "10:58": (100.0, 101.5, 100.0, 101.5)})  # bucket closes 11:00
    t = only_trade(run(ok))
    assert t.entry_time.strftime("%H:%M") == "11:00"


def test_cutoff_must_follow_range():
    with pytest.raises(ParamError):
        StrategyParams(orb_start="10:00", range_minutes=60, cutoff="10:30").validate()


def test_one_trade_per_day_and_reentry_policies():
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:10": (101.5, 104.25, 101.5, 104.0)})
    assert len(run(bars).trades) == 1
    two = run(bars, params=StrategyParams(**{**BASE.to_dict(), "max_trades": 2, "reentry": "any"})).trades
    assert len(two) == 2
    assert two.iloc[1].entry_time.strftime("%H:%M") == "10:15"  # first bucket closing after the 10:10 exit
    assert two.iloc[1].entry_price == 104.0
    after_loss_only = run(bars, params=StrategyParams(**{**BASE.to_dict(), "max_trades": 2, "reentry": "either_after_loss"}))
    assert len(after_loss_only.trades) == 1  # first trade was a winner


def test_opposite_reentry_after_stop():
    bars = day_bars(
        DATE,
        {**LONG_SIGNAL, "10:10": (101.5, 101.5, 98.75, 98.75), "10:14": (98.75, 98.75, 98.5, 98.5)},
    )
    p = StrategyParams(**{**BASE.to_dict(), "max_trades": 2, "reentry": "opposite_after_loss"})
    trades = run(bars, params=p).trades
    assert list(trades.direction) == ["long", "short"]
    p_same = StrategyParams(**{**BASE.to_dict(), "max_trades": 2, "reentry": "same_after_loss"})
    assert list(run(bars, params=p_same).trades.direction) == ["long"]


def test_direction_filter():
    bars = day_bars(DATE, {**SHORT_SIGNAL})
    assert len(run(bars, params=StrategyParams(**{**BASE.to_dict(), "direction": "long"})).trades) == 0
    assert len(run(bars, params=StrategyParams(**{**BASE.to_dict(), "direction": "short"})).trades) == 1


def test_commission_and_slippage_accounting():
    inst = make_instrument(commission_per_side=1.0, exchange_fee_per_side=0.5, slippage_ticks=1.0)
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:10": (101.5, 104.75, 101.5, 104.5)})
    t = only_trade(run(bars, instrument=inst))
    assert t.entry_price == 101.75  # 1 tick slippage on the market entry
    assert t.risk_ticks == 11 and t.target_price == 104.5
    assert t.exit_price == 104.5  # limit target: no slippage
    assert t.gross_pnl == pytest.approx(3.0 * 50)
    assert t.slippage_cost == pytest.approx(12.5)
    assert t.commission == pytest.approx(2.0)  # $1/contract/side x 2 fills x 1 contract
    assert t.fees == pytest.approx(1.0)  # $0.50/contract/side x 2 fills
    assert t.total_cost == pytest.approx(12.5 + 3.0)
    assert t.net_pnl == pytest.approx(150 - 12.5 - 3.0)


def test_frictionless_benchmark_has_zero_costs():
    inst = make_instrument(commission_per_side=1.0, exchange_fee_per_side=0.5, slippage_ticks=2.0)
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:10": (101.5, 104.25, 101.5, 104.0)})
    t = only_trade(run(bars, instrument=inst, execution=ExecutionParams(frictionless=True)))
    assert t.entry_price == 101.5 and t.commission == 0 and t.slippage_cost == 0


def test_stop_exit_slippage():
    inst = make_instrument(slippage_ticks=2.0)
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:10": (102.0, 102.0, 98.0, 98.5)})
    t = only_trade(run(bars, instrument=inst))
    assert t.entry_price == 102.0 and t.stop_price == 99.0
    assert t.exit_price == 98.5  # stop 99.00 minus 2 ticks
    assert t.exit_slip_ticks == 2 and t.slippage_cost == pytest.approx(4 * 12.5)


@pytest.mark.parametrize(
    "sizing,qty",
    [
        (SizingParams(mode="fixed_contracts", contracts=3), 3),
        (SizingParams(mode="fixed_risk", risk_dollars=1000), 8),  # 1000 / (10 ticks * 12.5)
        (SizingParams(mode="pct_equity", risk_pct=0.01, starting_equity=100_000), 8),
        (SizingParams(mode="fixed_risk", risk_dollars=100), 0),  # cannot afford 1 contract -> skipped
    ],
)
def test_position_sizing(sizing, qty):
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:10": (101.5, 104.25, 101.5, 104.0)})
    t = only_trade(run(bars, sizing=sizing))
    assert t.qty == qty
    assert bool(t.sized_out) == (qty == 0)
    assert t.net_pnl == pytest.approx(125.0 * qty)


def test_pct_equity_compounds():
    frames = [day_bars(d, {**LONG_SIGNAL, "10:10": (101.5, 104.25, 101.5, 104.0)}) for d in ("2024-01-03", "2024-01-04")]
    res = run(frames, sizing=SizingParams(mode="pct_equity", risk_pct=0.05, starting_equity=10_000))
    # 5% of 10,000 = 500 -> 4 contracts; after +500, 5% of 10,500 = 525 -> still 4
    assert list(res.trades.qty) == [4, 4]
    assert res.trades.equity_after.iloc[-1] == pytest.approx(11_000)


def test_time_exit():
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:59": (101.5, 102.0, 101.5, 102.0)})
    p = StrategyParams(**{**BASE.to_dict(), "time_exit": "11:00"})
    t = only_trade(run(bars, params=p))
    assert t.exit_reason == "time" and t.exit_time.strftime("%H:%M") == "11:00" and t.exit_price == 102.0


def test_stop_entry_intrabar_and_gap():
    p = StrategyParams(**{**BASE.to_dict(), "entry_method": "stop"})
    bars = day_bars(DATE, {**RANGE_BARS, "09:50": (100.0, 101.25, 100.0, 101.0)})
    t = only_trade(run(bars, params=p))
    assert t.entry_price == 101.25 and t.entry_time.strftime("%H:%M") == "09:50" and t.intrabar_entry
    assert t.risk_ticks == 9 and t.target_price == 103.5
    gap = day_bars(DATE, {**RANGE_BARS, "09:49": (100, 100.75, 100, 100.75), "09:50": (101.5, 101.75, 101.5, 101.5)})
    t = only_trade(run(gap, params=p))
    assert t.entry_price == 101.5 and not t.intrabar_entry


def test_stop_entry_confirmation_offset_and_buffer():
    # offset 1 tick -> level 101.25 -> first price strictly above is 101.50; +2 buffer ticks -> 102.00
    p = StrategyParams(**{**BASE.to_dict(), "entry_method": "stop", "confirm_ticks": 1, "entry_buffer_ticks": 2})
    bars = day_bars(DATE, {**RANGE_BARS, "09:50": (100.0, 101.75, 100.0, 101.5), "09:51": (101.5, 102.25, 101.5, 102.0)})
    t = only_trade(run(bars, params=p))
    assert t.entry_price == 102.0 and t.entry_time.strftime("%H:%M") == "09:51"


def test_stop_entry_both_sides_same_bar_is_skipped():
    p = StrategyParams(**{**BASE.to_dict(), "entry_method": "stop"})
    bars = day_bars(DATE, {**RANGE_BARS, "09:50": (100.0, 104.0, 98.0, 100.0)})
    res = run(bars, params=p)
    assert len(res.trades) == 0
    assert res.day_status.status.iloc[0] == "ambiguous_both_sides_entry"


def test_intrabar_entry_bar_target_rules():
    p = StrategyParams(**{**BASE.to_dict(), "entry_method": "stop"})
    bars = day_bars(DATE, {**RANGE_BARS, "09:50": (100.0, 104.0, 100.0, 103.75)})
    pess = only_trade(run(bars, params=p, execution=ExecutionParams(ambiguity="pessimistic")))
    assert pess.exit_reason == "target" and pess.exit_time == pess.entry_time
    cons = only_trade(run(bars, params=p, execution=EXEC))
    assert cons.exit_time > cons.entry_time  # conservative: no target on the fill bar


def test_limit_entry_standard_vs_conservative():
    p = StrategyParams(**{**BASE.to_dict(), "entry_method": "limit"})
    bars = day_bars(
        DATE,
        {**LONG_SIGNAL, "10:00": (101.5, 101.5, 101.0, 101.25), "10:05": (101.25, 101.25, 100.75, 101.0)},
    )
    std = only_trade(run(bars, params=p, execution=ExecutionParams(fill_model="standard")))
    assert std.entry_price == 101.0 and std.entry_time.strftime("%H:%M") == "10:00" and std.entry_slip_ticks == 0
    cons = only_trade(run(bars, params=p, execution=EXEC))
    assert cons.entry_price == 101.0 and cons.entry_time.strftime("%H:%M") == "10:05"
    never = day_bars(DATE, {**LONG_SIGNAL, "10:00": (101.5, 101.5, 101.0, 101.25)})
    res = run(never, params=p, execution=EXEC)
    assert len(res.trades) == 0 and res.day_status.status.iloc[0] == "signal_not_filled"


def test_stop_methods():
    bars = day_bars(DATE, {**LONG_SIGNAL, "10:10": (101.5, 110.0, 101.5, 110.0)})
    mid = only_trade(run(bars, params=StrategyParams(**{**BASE.to_dict(), "stop_method": "or_mid"})))
    assert mid.stop_price == 100.0 and mid.target_price == 103.0
    pct = only_trade(run(bars, params=StrategyParams(**{**BASE.to_dict(), "stop_method": "or_pct", "stop_param": 0.5})))
    assert pct.stop_price == 100.5  # 101.5 - 0.5 * 2.0
    fixed = only_trade(run(bars, params=StrategyParams(**{**BASE.to_dict(), "stop_method": "fixed_ticks", "stop_param": 6})))
    assert fixed.stop_price == 100.0 and fixed.risk_ticks == 6
    # 33% of width = 0.66 points = 2.64 ticks -> rounded away from entry to 3 ticks
    third = only_trade(run(bars, params=StrategyParams(**{**BASE.to_dict(), "stop_method": "or_pct", "stop_param": 0.33})))
    assert third.risk_ticks == 3


def test_atr_stop_requires_history():
    frames = [day_bars(d, {**LONG_SIGNAL, "10:10": (101.5, 110.0, 101.5, 110.0)}) for d in pd.bdate_range("2024-01-02", "2024-01-12").strftime("%Y-%m-%d")]
    p = StrategyParams(**{**BASE.to_dict(), "stop_method": "atr", "stop_param": 1.0, "atr_period": 3})
    res = run(frames, params=p)
    statuses = list(res.day_status.status)
    assert statuses[:3] == ["atr_unavailable"] * 3  # needs 3 prior sessions
    assert statuses[3] == "traded"


def test_breakeven_stop():
    bars = day_bars(
        DATE,
        {**LONG_SIGNAL, "10:00": (101.5, 103.0, 101.5, 102.5), "10:05": (102.5, 102.5, 101.25, 101.25)},
    )
    p = StrategyParams(**{**BASE.to_dict(), "breakeven_r": 0.5})
    t = only_trade(run(bars, params=p))
    assert t.exit_reason == "breakeven_stop" and t.exit_price == 101.5 and t.net_pnl == 0.0


def test_or_atr_filter():
    frames = [day_bars(d, dict(LONG_SIGNAL)) for d in pd.bdate_range("2024-01-02", "2024-01-10").strftime("%Y-%m-%d")]
    p = StrategyParams(**{**BASE.to_dict(), "or_atr_min": 5.0, "atr_period": 2})
    res = run(frames, params=p)
    assert "filtered_or_atr" in set(res.day_status.status)
    assert len(res.trades) == 0


def test_identical_results_across_dst():
    """Same ET wall-clock path in EST (January) and EDT (July) must trade identically."""
    spec = {**LONG_SIGNAL, "10:10": (101.5, 104.25, 101.5, 104.0)}
    winter = only_trade(run(day_bars("2024-01-03", spec)))
    summer = only_trade(run(day_bars("2024-07-03", spec)))
    assert winter.entry_time.strftime("%H:%M") == summer.entry_time.strftime("%H:%M") == "09:55"
    assert winter.net_pnl == summer.net_pnl
    assert winter.entry_time.tz_convert("UTC").hour == 14 and summer.entry_time.tz_convert("UTC").hour == 13


def test_different_range_and_entry_timeframes():
    # 10-minute entry buckets aligned to the range end: [09:45, 09:55) closes 09:55
    bars = day_bars(DATE, {**RANGE_BARS, "09:52": (100.0, 101.5, 100.0, 101.5)})
    t = only_trade(run(bars, params=StrategyParams(**{**BASE.to_dict(), "entry_tf": 10})))
    assert t.entry_time.strftime("%H:%M") == "09:55"
    t = only_trade(run(bars, params=StrategyParams(**{**BASE.to_dict(), "entry_tf": 1})))
    assert t.entry_time.strftime("%H:%M") == "09:53"
    # with a 30-minute range (09:30-10:00) the 09:52 move is inside the range: no trade
    assert len(run(bars, params=StrategyParams(**{**BASE.to_dict(), "range_minutes": 30})).trades) == 0


def test_confirmation_thresholds():
    bars = day_bars(DATE, {**RANGE_BARS, "09:54": (100.0, 101.25, 100.0, 101.25)})  # close 1 tick above
    assert len(run(bars).trades) == 1
    assert len(run(bars, params=StrategyParams(**{**BASE.to_dict(), "confirm_ticks": 1})).trades) == 0
    # 5% of 2.0 width = 0.1 -> close must exceed 101.10: 101.25 qualifies; 10% -> 101.20 qualifies too
    assert len(run(bars, params=StrategyParams(**{**BASE.to_dict(), "confirm_or_frac": 0.10})).trades) == 1
    assert len(run(bars, params=StrategyParams(**{**BASE.to_dict(), "confirm_or_frac": 0.15})).trades) == 0
