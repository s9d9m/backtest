"""PHASE0_FREE_PROXY: Yahoo handling, ETF specifics, split isolation and final-test leakage (no network)."""

import json

import numpy as np
import pandas as pd
import pytest

from orb_lab.data_sources import yahoo
from orb_lab.engine.backtest import run_backtest
from orb_lab.engine.params import ExecutionParams, SizingParams, StrategyParams
from orb_lab.engine.pipeline import build_dataset
from orb_lab.optimization.parameter_space import ParameterSpace
from orb_lab.optimization.robustness import neighbour_index
from orb_lab.optimization.walk_forward import SelectionConfig
from orb_lab.phase0 import pipeline as p0
from orb_lab.phase0.controls import both_direction_outcomes, random_direction_control

CFG = p0.load_config()
NY = "America/New_York"


def yahoo_frame(dates, paths=None, seed=0):
    """yfinance-style 5-minute RTH frame (index in America/New_York, capitalised columns)."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 500.0
    for d in dates:
        for k, t in enumerate(pd.date_range(f"{d} 09:30", f"{d} 15:55", freq="5min")):
            if paths and (str(d), t.strftime("%H:%M")) in paths:
                o, h, l, c = paths[(str(d), t.strftime("%H:%M"))]
            else:
                o = price
                c = round(o + rng.normal(0, 0.3), 2)
                h, l = round(max(o, c) + abs(rng.normal(0, 0.1)), 2), round(min(o, c) - abs(rng.normal(0, 0.1)), 2)
            rows.append((t, o, h, l, c, 1000 + (50000 if k == 0 else 0)))
            price = c
    f = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"]).set_index("ts")
    f.index = pd.DatetimeIndex(f.index).tz_localize(NY)
    f["Adj Close"] = f["Close"]
    return f


class FakeTicker:
    def __init__(self, frame, limit_days=None):
        self.frame, self.calls = frame, []

    def history(self, start, end, interval, prepost, auto_adjust, actions, raise_errors):
        self.calls.append((start, end, interval))
        s, e = pd.Timestamp(start).tz_localize(NY), pd.Timestamp(end).tz_localize(NY)
        return self.frame[(self.frame.index >= s) & (self.frame.index < e)]


def test_yahoo_download_timestamps_cache_and_provenance(tmp_path):
    dates = ["2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12"]  # spans the 2024-03-10 DST change
    tk = FakeTicker(yahoo_frame(dates))
    prov = yahoo.download("SPY", "5m", "2024-03-07", "2024-03-13", chunk_days=3, ticker=tk, data_dir=tmp_path)
    assert prov["rows_raw"] == 4 * 78 and prov["source_timezone"] == NY and prov["interval"] == "5m"
    assert prov["label"] == yahoo.PHASE0_LABEL and prov["experiment_family"] == "PHASE0_FREE_PROXY"
    out = pd.read_parquet(tmp_path / "phase0" / "SPY_5m.parquet")
    assert str(out["timestamp"].dt.tz) == "UTC"
    # 09:30 EST = 14:30 UTC before the change, 09:30 EDT = 13:30 UTC after it
    first = out.groupby(out["timestamp"].dt.date)["timestamp"].min()
    assert [t.hour for t in first] == [14, 14, 13, 13]
    n_calls = len(tk.calls)
    yahoo.download("SPY", "5m", "2024-03-07", "2024-03-13", chunk_days=3, ticker=FakeTicker(yahoo_frame([])), data_dir=tmp_path)
    ds = build_dataset(tmp_path / "phase0" / "SPY_5m.parquet", p0.etf_instrument("SPY", CFG, 0.02))
    assert ds.prep.base_minutes == 5 and set(ds.prep.tod[ds.prep.day_start]) == {570}
    assert n_calls == 2


def test_yahoo_refusal_is_reported_not_hidden(tmp_path):
    class Refusing:
        def history(self, **kw):
            raise RuntimeError("1m data not available ... must be within the last 30 days")

    with pytest.raises(yahoo.YahooUnavailable, match="within the last 30 days"):
        yahoo.download("SPY", "1m", "2020-01-01", "2020-01-10", chunk_days=7, ticker=Refusing(), data_dir=tmp_path)


def test_interval_choice_prefers_finest_with_enough_history():
    avail = {"1m": {"sessions": 20}, "2m": {"sessions": 31}, "5m": {"sessions": 42}, "15m": {"sessions": 42}}
    assert p0.choose_interval(avail) == "5m"
    assert p0.choose_interval({**avail, "1m": {"sessions": 40}}) == "1m"
    assert p0.choose_interval({"1m": {"sessions": 5}, "5m": {"sessions": 10}}) == "5m"


def _dataset(tmp_path, dates, paths=None, name="SPY", seed=0):
    f = yahoo_frame(dates, paths, seed)
    prov = yahoo.download(name, "5m", dates[0], str((pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)).date()), chunk_days=40,
                          ticker=FakeTicker(f), data_dir=tmp_path)
    return tmp_path / "phase0" / f"{name}_5m.parquet"


def test_etf_spec_and_space_dedup():
    inst = p0.etf_instrument("SPY", CFG, 0.02)
    assert inst.tick_size == 0.01 and inst.tick_value == 0.01 and inst.multiplier == 1
    assert inst.commission_per_side == 0.02 and inst.slippage_ticks == 0
    configs, info = p0.build_space(CFG, 5)
    assert all(c.entry_tf in (0, 5, 10, 15) for c in configs)
    assert not any(c.entry_method == "limit" and c.stop_method == "or_pct" and c.stop_param == 0.5 for c in configs)
    assert info["evaluated"] == len(configs) == len({c.key() for c in configs})
    assert info["equivalent_limit_or_pct50_removed"] > 0


def test_or_construction_entry_and_costs_on_5m_etf_bars(tmp_path):
    d = "2024-06-04"
    paths = {
        (d, "09:30"): (500.00, 500.50, 499.80, 500.20),
        (d, "09:35"): (500.20, 500.60, 499.90, 500.40),   # 10-minute OR: high 500.60, low 499.80
        (d, "09:40"): (500.40, 500.70, 500.30, 500.65),   # closes above 500.60 -> signal at 09:45
        (d, "09:45"): (500.66, 500.70, 500.60, 500.68),   # market entry at this open
        (d, "09:50"): (500.68, 501.20, 500.60, 501.10),   # 1R target = 500.66 + 0.86 = 501.52 not reached
        (d, "09:55"): (501.10, 501.60, 501.00, 501.50),   # trades through 501.52 -> target
    }
    path = _dataset(tmp_path, [d], paths)
    inst = p0.etf_instrument("SPY", CFG, 0.02)
    ds = build_dataset(path, inst)
    hi, lo, _, ok = ds.prep.opening_range(570, 10)
    assert ok[0] and hi[0] * 0.01 == pytest.approx(500.60) and lo[0] * 0.01 == pytest.approx(499.80)
    params = StrategyParams(range_minutes=10, entry_tf=5, stop_method="or_opposite", target_r=1.0, cutoff="11:00")
    t = run_backtest(ds.prep, params, ExecutionParams(), SizingParams(contracts=100)).trades.iloc[0]
    assert t.entry_time.strftime("%H:%M") == "09:45" and t.entry_price == pytest.approx(500.66)
    assert t.stop_price == pytest.approx(499.80) and t.target_price == pytest.approx(501.52)
    assert t.exit_reason == "target" and t.exit_time.strftime("%H:%M") == "09:55"
    # $0.02/share on every fill: 100 shares x 2 fills x $0.02 = $4
    assert t.commission == pytest.approx(4.0) and t.gross_pnl == pytest.approx(86.0) and t.net_pnl == pytest.approx(82.0)
    assert t.r_multiple == pytest.approx((0.86 - 0.04) / 0.86)
    # one ETF tick of confirmation on a stop entry: trigger strictly above 500.61 -> 500.62
    stop_params = StrategyParams(range_minutes=10, entry_method="stop", confirm_ticks=1, stop_method="or_opposite", target_r=1.0, cutoff="11:00")
    ts = run_backtest(ds.prep, stop_params, ExecutionParams(), SizingParams(contracts=100)).trades.iloc[0]
    assert ts.entry_price == pytest.approx(500.62)


def test_missing_bars_and_misaligned_open_are_caught(tmp_path):
    dates = ["2024-06-03", "2024-06-04", "2024-06-05"]
    f = yahoo_frame(dates)
    f = f.drop(f.index[100])  # one missing 5m bar on 06-04
    inst = p0.etf_instrument("SPY", CFG, 0.02)
    path = tmp_path / "a.parquet"
    pd.DataFrame({"timestamp": f.index.tz_convert("UTC"), "open": f.Open, "high": f.High, "low": f.Low, "close": f.Close, "volume": f.Volume}).to_parquet(path)
    _, dq = p0.phase0_dq("SPY", path, inst, 5, None)
    assert dq["summary"]["sessions_incomplete"] == 1 and dq["summary"]["verdict"] in ("REVIEW", "STOP")
    g = yahoo_frame(dates)
    g = g[~((g.index.date == pd.Timestamp("2024-06-05").date()) & (g.index.strftime("%H:%M") == "09:30"))]
    path2 = tmp_path / "b.parquet"
    pd.DataFrame({"timestamp": g.index.tz_convert("UTC"), "open": g.Open, "high": g.High, "low": g.Low, "close": g.Close, "volume": g.Volume}).to_parquet(path2)
    _, dq2 = p0.phase0_dq("SPY", path2, inst, 5, None)
    assert dq2["summary"]["verdict"] == "STOP" and dq2["summary"]["sessions_first_bar_not_0930"] == 1


def test_split_is_chronological():
    assert p0.split_sessions(42, CFG["split"]) == (25, 8, 9)
    assert p0.split_sessions(10, CFG["split"]) == (6, 2, 2)
    with pytest.raises(ValueError):
        p0.split_sessions(2, CFG["split"])


SMALL = ParameterSpace("t", {"range_minutes": [10, 15, 30], "entry_tf": [5, 10], "entry_method": ["market", "limit"],
                             "stop": ["or_mid", "or_opposite"], "target_r": [0.75, 1.0, 1.5], "cutoff": ["11:00", "12:00"],
                             "direction": ["both", "long"]})


def test_selection_never_sees_the_final_test(tmp_path):
    dates = [str(d.date()) for d in pd.bdate_range("2024-06-03", "2024-08-02")]
    dates = [d for d in dates if d not in ("2024-06-19", "2024-07-04")]
    inst = p0.etf_instrument("SPY", CFG, 0.02)
    prep = build_dataset(_dataset(tmp_path / "a", dates, seed=3), inst).prep
    n_train, n_val, n_test = p0.split_sessions(prep.n_days, CFG["split"])
    configs = SMALL.expand(5).configs
    nbrs = neighbour_index(configs)
    sel = SelectionConfig(n_finalists=10, min_trades_train=5, min_trades_val=2)
    a = p0.select_without_test(prep, configs, nbrs, n_train, n_val, sel, ExecutionParams())
    assert a["dev_prep"].n_days == n_train + n_val and a["dev_prep"].dates[-1] < prep.dates[n_train + n_val]
    # replace every final-test bar with a completely different market: the selection must not change
    test_start = pd.Timestamp(prep.dates[n_train + n_val])
    other = yahoo_frame(dates, seed=99)
    base = yahoo_frame(dates, seed=3)
    mixed = pd.concat([base[base.index.normalize().tz_localize(None) < test_start], other[other.index.normalize().tz_localize(None) >= test_start]])
    prov = yahoo.download("SPY", "5m", dates[0], "2024-08-03", chunk_days=90, ticker=FakeTicker(mixed), data_dir=tmp_path / "b")
    prep_b = build_dataset(tmp_path / "b" / "phase0" / "SPY_5m.parquet", inst).prep
    assert not np.array_equal(prep.close, prep_b.close)
    b = p0.select_without_test(prep_b, configs, nbrs, n_train, n_val, sel, ExecutionParams())
    assert a["pick"]["selected"] == b["pick"]["selected"]
    pd.testing.assert_frame_equal(a["pick"]["finalists"], b["pick"]["finalists"])
    assert a["best_in_sample"] == b["best_in_sample"] and a["best_dev_pnl"] == b["best_dev_pnl"]


def test_trade_log_columns_and_control_reproduces_engine(tmp_path):
    dates = [str(d.date()) for d in pd.bdate_range("2024-06-03", "2024-06-28") if str(d.date()) != "2024-06-19"]
    inst = p0.etf_instrument("SPY", CFG, 0.02)
    prep = build_dataset(_dataset(tmp_path, dates, seed=5), inst).prep
    params = StrategyParams(range_minutes=10, entry_tf=5, entry_method="limit", stop_method="or_opposite", target_r=1.0, cutoff="12:00")
    r = p0.split_result(prep, params, inst, 0.02, 100, pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]))
    t = r["_trades"]
    assert len(t) > 3
    log = t.assign(split="train", symbol="SPY", estimated_friction=t["slippage_cost"] + t["commission"])
    for col in p0.TRADE_LOG_COLUMNS:
        assert col in log.columns, col
    assert (log["estimated_friction"] == 4.0).all()
    assert r["sum_r_without_best_1"] <= r["sum_r"]
    oc = both_direction_outcomes(prep, t, params.target_r, 0.0, prep.session_exit_min, cost_ticks_round_trip=4.0)
    assert np.allclose(oc["actual_r_sim"].to_numpy(), t["r_multiple"].to_numpy(), atol=1e-9)
    ctrl = random_direction_control(oc, n_sims=2000, seed=1)
    assert 0.0 <= ctrl["p_value_random_ge_strategy"] <= 1.0 and ctrl["n_trades"] == len(t)
    # friction reduces every trade's R by exactly 2*friction/risk
    r0 = p0.split_result(prep, params, inst, 0.0, 100, pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]))["_trades"]
    assert np.allclose(r0["r_multiple"] - t["r_multiple"], 0.04 / (t["risk_ticks"] * 0.01))
