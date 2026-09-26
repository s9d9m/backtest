"""Tests that actively try to detect future leakage.

Core property: at time T the strategy may only use information known at or before T. Therefore
deleting or altering data *after* T must not change any decision made before T.
"""

import numpy as np
import pandas as pd
import pytest

from orb_lab.engine.backtest import run_backtest
from orb_lab.engine.data_quality import check_and_clean
from orb_lab.engine.orb import trailing_percentile_rank, wilder_atr_known_before
from orb_lab.engine.params import ExecutionParams, SizingParams, StrategyParams
from orb_lab.engine.session_data import prepare_data
from orb_lab.engine.sessions import TradingCalendar, assign_sessions
from orb_lab.engine.synthetic import generate_bars

from .conftest import RANGE_BARS, day_bars, make_instrument, prepare

CAL = TradingCalendar("WEEKDAYS")
INST = make_instrument(slippage_ticks=1.0, commission_per_side=1.0, exchange_fee_per_side=1.0, tick_size=0.25, tick_value=12.5, multiplier=50)


@pytest.fixture(scope="module")
def synthetic_bars():
    inst = make_instrument(symbol="ES", slippage_ticks=1.0)
    return generate_bars(inst, "2023-01-02", "2023-04-28", seed=11, calendar="WEEKDAYS", trend_strength=0.5)


def _prep(bars):
    clean, report = check_and_clean(bars, INST, calendar=CAL)
    assert report.ok, report.to_frame()
    return prepare_data(clean, INST, calendar=CAL)


def _entries(trades, before=None, upto_day=None):
    t = trades
    if upto_day is not None:
        t = t[t.session_date <= upto_day]
    cols = ["session_date", "direction", "entry_time", "entry_price", "stop_price", "target_price"]
    return t[cols].reset_index(drop=True)


PARAM_SETS = [
    StrategyParams(entry_method="market", range_minutes=15, entry_tf=5, target_r=1.0, cutoff="12:00", stop_method="atr", stop_param=1.0, atr_period=5),
    StrategyParams(entry_method="limit", range_minutes=10, entry_tf=3, entry_buffer_ticks=1, target_r=1.5, cutoff="12:00"),
    StrategyParams(entry_method="stop", range_minutes=20, target_r=0.8, cutoff="11:30", stop_method="or_mid"),
    StrategyParams(entry_method="market", range_minutes=30, entry_tf=15, max_trades=0, target_r=1.2, cutoff="13:00", or_atr_max=2.0, atr_period=5),
]


@pytest.mark.parametrize("params", PARAM_SETS, ids=["market_atr", "limit", "stop", "unlimited_orfilter"])
def test_truncating_a_session_never_changes_earlier_decisions(synthetic_bars, params):
    """Remove all data of one session after time T; every decision before T (and on earlier days) must be identical."""
    full = run_backtest(_prep(synthetic_bars), params, ExecutionParams())
    assert len(full.trades) > 20
    session_date, tod = assign_sessions(synthetic_bars.index)
    rng = np.random.default_rng(0)
    days = full.trades.session_date.dt.date.unique()
    for day in rng.choice(days, size=min(6, len(days)), replace=False):
        cut_tod = int(rng.integers(600, 780))
        keep = ~((session_date == np.datetime64(day)) & (tod >= cut_tod))
        truncated = run_backtest(_prep(synthetic_bars[keep]), params, ExecutionParams())
        cut_time = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(minutes=cut_tod)
        a = full.trades[(full.trades.session_date.dt.date < day) | (full.trades.entry_time < cut_time)]
        b = truncated.trades[(truncated.trades.session_date.dt.date < day) | (truncated.trades.entry_time < cut_time)]
        pd.testing.assert_frame_equal(_entries(a), _entries(b))
        # exits completed before the cut are unchanged too
        done_a = a[a.exit_time < cut_time - pd.Timedelta(minutes=1)][["entry_time", "exit_time", "exit_price", "exit_reason"]]
        done_b = b[b.exit_time < cut_time - pd.Timedelta(minutes=1)][["entry_time", "exit_time", "exit_price", "exit_reason"]]
        pd.testing.assert_frame_equal(done_a.reset_index(drop=True), done_b.reset_index(drop=True))


def test_future_days_do_not_change_past_trades(synthetic_bars):
    params = PARAM_SETS[0]
    full = run_backtest(_prep(synthetic_bars), params)
    cutoff = pd.Timestamp("2023-03-01", tz="UTC")
    head = run_backtest(_prep(synthetic_bars[synthetic_bars.index < cutoff]), params)
    a = full.trades[full.trades.session_date < "2023-02-28"]
    b = head.trades[head.trades.session_date < "2023-02-28"]
    pd.testing.assert_frame_equal(_entries(a), _entries(b))
    assert (a.net_pnl.to_numpy() == b.net_pnl.to_numpy()).all()


def test_signal_bar_contents_after_fill_open_do_not_change_entry():
    base = {**RANGE_BARS, "09:54": (100.0, 101.5, 100.0, 101.5), "09:55": (101.5, 101.75, 101.25, 101.5)}
    altered = {**RANGE_BARS, "09:54": (100.0, 101.5, 100.0, 101.5), "09:55": (101.5, 103.0, 100.0, 100.25)}
    p = StrategyParams(target_r=5.0)
    a = run_backtest(prepare(day_bars("2024-01-03", base)), p).trades.iloc[0]
    b = run_backtest(prepare(day_bars("2024-01-03", altered)), p).trades.iloc[0]
    assert a.entry_price == b.entry_price == 101.5 and a.entry_time == b.entry_time


def test_breakout_is_not_acted_on_before_bucket_close():
    # The 09:50 bar closes above the range, but the 5-minute bucket [09:50,09:55) closes back inside.
    bars = day_bars("2024-01-03", {**RANGE_BARS, "09:50": (100.0, 101.5, 100.0, 101.5), "09:54": (101.5, 101.5, 100.5, 100.5)})
    assert len(run_backtest(prepare(bars), StrategyParams()).trades) == 0


def test_opening_range_ignores_bars_after_window():
    a = prepare(day_bars("2024-01-03", RANGE_BARS)).opening_range(570, 15)
    b = prepare(day_bars("2024-01-03", {**RANGE_BARS, "09:45": (100, 120, 80, 100)})).opening_range(570, 15)
    assert a[0][0] == b[0][0] and a[1][0] == b[1][0]


def test_atr_uses_only_prior_sessions():
    rng = np.random.default_rng(3)
    high = 100 + rng.random(50) * 5
    low = high - 1 - rng.random(50) * 3
    close = (high + low) / 2
    base = wilder_atr_known_before(high, low, close, 14)
    high2 = high.copy()
    high2[30:] += 50  # change session 30 and later
    changed = wilder_atr_known_before(high2, low, close, 14)
    np.testing.assert_array_equal(base[:31], changed[:31])  # ATR for session 30 is known before 30 trades
    assert changed[31] != base[31]


def test_trailing_percentile_rank_is_causal():
    values = np.random.default_rng(1).random(200)
    base = trailing_percentile_rank(values, lookback=60, min_history=20)
    future = values.copy()
    future[150:] = 1000.0
    changed = trailing_percentile_rank(future, lookback=60, min_history=20)
    np.testing.assert_array_equal(base[:150], changed[:150])
    assert np.isnan(base[:20]).all()


@pytest.mark.parametrize("params", [PARAM_SETS[0], PARAM_SETS[3]], ids=["market_atr", "unlimited_orfilter"])
def test_scrambling_everything_after_an_open_never_changes_market_entries_at_that_open(synthetic_bars, params):
    """Keep bar T's open, scramble its high/low/close and all later bars of the session.

    A market entry filled at T's open may only depend on information up to T's open, so every entry
    with ``entry_time <= T`` must be identical (this also catches peeking at the fill bar's close).
    """
    full = run_backtest(_prep(synthetic_bars), params)
    session_date, tod = assign_sessions(synthetic_bars.index)
    rng = np.random.default_rng(5)
    for _, trade in full.trades.sample(8, random_state=1).iterrows():
        day = trade.session_date.date()
        cut_tod = int(trade.entry_time.hour * 60 + trade.entry_time.minute)
        on_day = session_date == np.datetime64(day)
        after = on_day & (tod >= cut_tod)
        scrambled = synthetic_bars.copy()
        n = int(after.sum())
        opens = scrambled.loc[after, "open"].to_numpy().copy()
        shock = np.round(rng.normal(0, 40, n)) * 0.25
        new_close = opens + shock
        spread = np.abs(np.round(rng.normal(0, 10, n)) * 0.25)
        scrambled.loc[after, "close"] = new_close
        scrambled.loc[after, "high"] = np.maximum(opens, new_close) + spread
        scrambled.loc[after, "low"] = np.minimum(opens, new_close) - spread
        idx = np.where(after)[0]
        # later bars get fresh opens too (only bar T keeps its original open)
        scrambled.iloc[idx[1:], scrambled.columns.get_loc("open")] = new_close[:-1]
        scrambled.loc[after, "high"] = scrambled.loc[after, ["open", "high", "close"]].max(axis=1)
        scrambled.loc[after, "low"] = scrambled.loc[after, ["open", "low", "close"]].min(axis=1)
        other = run_backtest(_prep(scrambled), params)
        limit = trade.entry_time
        a = full.trades[(full.trades.session_date.dt.date < day) | (full.trades.entry_time <= limit)]
        b = other.trades[(other.trades.session_date.dt.date < day) | (other.trades.entry_time <= limit)]
        pd.testing.assert_frame_equal(_entries(a), _entries(b))
