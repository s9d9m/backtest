"""Streamlit dashboard: DATA | BACKTEST | OPTIMIZATION | WALK-FORWARD | ROBUSTNESS | MONTE CARLO | TRADE LOG | REPORT."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .. import __version__
from ..engine.backtest import run_backtest
from ..engine.data_quality import CleaningPolicy
from ..engine.instruments import load_instruments
from ..engine.params import (
    AMBIGUITY_MODES,
    DIRECTIONS,
    ENTRY_METHODS,
    FILL_MODELS,
    REENTRY_POLICIES,
    SIZING_MODES,
    STOP_METHODS,
    ExecutionParams,
    SizingParams,
    StrategyParams,
    load_strategy_config,
)
from ..engine.pipeline import DataQualityError, build_dataset
from ..engine.synthetic import generate_bars
from ..optimization.grid_search import results_table, run_grid
from ..optimization.heatmaps import AGGREGATIONS, STANDARD_PAIRS, heatmap_table, varying_parameters
from ..optimization.objectives import ALTERNATIVE_OBJECTIVES, ObjectiveConfig
from ..optimization.parameter_space import CONFIRMATION_PRESETS, ParameterSpace, load_search_spaces
from ..reports import plots
from ..reports.experiment import metrics_table, save_backtest

TIMES_ORB = ["09:00", "09:15", "09:30", "09:45", "10:00"]
RANGES = [5, 10, 15, 20, 30, 45, 60]
ENTRY_TFS = [1, 2, 3, 5, 10, 15]
TARGETS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
CUTOFFS = ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "14:00", "15:00"]
TIME_EXITS = ["session", "10:30", "11:00", "11:30", "12:00", "13:00", "14:00", "15:00", "15:45", "15:55"]
STOP_CHOICES = ["or_opposite", "or_mid", "or_pct:0.25", "or_pct:0.33", "or_pct:0.5", "or_pct:0.67", "or_pct:0.75", "or_pct:1.0",
                "atr:0.5", "atr:0.75", "atr:1.0", "atr:1.25", "atr:1.5", "atr:2.0"]
SLIPPAGE_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def _state():
    return st.session_state


# ------------------------------------------------------------------------------------------ sidebar
def sidebar():
    st.sidebar.title("ORB Lab")
    st.sidebar.caption(f"v{__version__} · research platform, not a signal service")
    instruments = load_instruments()
    symbol = st.sidebar.selectbox("Market", list(instruments), key="symbol")
    inst = instruments[symbol]
    st.sidebar.caption(f"{inst.name} · tick {inst.tick_size} = ${inst.tick_value} · default slippage {inst.slippage_ticks} tick(s)")
    source = st.sidebar.radio("Data source", ["File path", "Upload", "Synthetic"], horizontal=True)
    tz = st.sidebar.selectbox("Timezone of naive timestamps", ["(timestamps carry offset)", "America/New_York", "America/Chicago", "UTC"])
    tz = None if tz.startswith("(") else tz
    convention = st.sidebar.selectbox("Timestamp marks", ["start", "end"], format_func=lambda x: f"bar {x}")
    with st.sidebar.expander("Cleaning policy"):
        dup = st.selectbox("Conflicting duplicates", ["error", "keep_first", "keep_last"])
        inv = st.selectbox("Invalid OHLC rows", ["error", "drop"])
        allow = st.checkbox("Allow unresolved errors (recorded)", value=False)
        coverage = st.slider("Min OR bar coverage", 0.5, 1.0, 0.8, 0.05)
        excl_early = st.checkbox("Exclude early-close sessions", value=False)
    src = None
    if source == "File path":
        default = next(iter(sorted(Path("data").glob("*.parquet"))), None) if Path("data").exists() else None
        src = st.sidebar.text_input("Path (CSV/Parquet)", value=str(default) if default else "")
    elif source == "Upload":
        up = st.sidebar.file_uploader("CSV or Parquet", type=["csv", "parquet", "pq", "txt"])
        if up is not None:
            tmp = Path(tempfile.gettempdir()) / f"orb_upload_{up.name}"
            tmp.write_bytes(up.getvalue())
            src = str(tmp)
    else:
        c1, c2 = st.sidebar.columns(2)
        s_start = c1.text_input("Start", "2019-01-01")
        s_end = c2.text_input("End", "2023-12-29")
        seed = st.sidebar.number_input("Seed", value=7, step=1)
        trend = st.sidebar.slider("Planted trend (daily σ; 0 = null model)", 0.0, 2.0, 0.0, 0.1)
    if st.sidebar.button("Load data", type="primary"):
        with st.spinner("Loading and checking data..."):
            try:
                if source == "Synthetic":
                    frame = generate_bars(inst, s_start, s_end, seed=int(seed), trend_strength=trend)
                    ds = build_dataset(frame, inst, min_or_coverage=coverage, exclude_early_close=excl_early)
                    ds.loaded.source = f"synthetic(seed={int(seed)}, trend={trend}, {s_start}..{s_end})"
                else:
                    if not src:
                        st.sidebar.error("Choose a file first")
                        return
                    ds = build_dataset(src, inst, source_tz=tz, timestamp_convention=convention,
                                       policy=CleaningPolicy(dup, inv), allow_errors=allow, min_or_coverage=coverage,
                                       exclude_early_close=excl_early)
                from ..research.lockbox import apply_lockbox

                ds.prep, lock = apply_lockbox(inst.symbol, ds.prep)
                if lock and lock.get("lockbox_start"):
                    st.sidebar.info(f"Lockbox: sessions from {lock['lockbox_start']} are withheld")
                _state()["dataset"] = ds
                _state()["instrument"] = inst
                for k in ("result", "grid"):
                    _state().pop(k, None)
                st.sidebar.success(f"{ds.prep.n_days:,} tradable sessions")
            except DataQualityError as exc:
                _state()["dq_block"] = exc.report
                st.sidebar.error("Data-quality errors block backtesting. See DATA tab.")
            except Exception as exc:  # surface loader errors to the user
                st.sidebar.error(str(exc))


# ------------------------------------------------------------------------------------------ DATA
def tab_data():
    ds = _state().get("dataset")
    blocked = _state().get("dq_block")
    if ds is None and blocked is None:
        st.info("Load a CSV/Parquet file or generate synthetic data from the sidebar. Required columns: "
                "timestamp, open, high, low, close, volume (optional bid, ask, contract, symbol).")
        return
    report = ds.report if ds is not None else blocked
    if not report.ok:
        st.error("Unresolved data-quality errors. Backtesting is blocked unless you choose an explicit cleaning policy "
                 "or allow errors (which is recorded in every manifest).")
    s = report.summary()
    c = st.columns(5)
    c[0].metric("Rows (raw)", f"{s['rows_raw']:,}")
    c[1].metric("Rows (clean)", f"{s['rows_clean']:,}")
    c[2].metric("Bar size", f"{s['bar_minutes']} min")
    c[3].metric("Errors", s["n_errors"])
    c[4].metric("Warnings", s["n_warnings"])
    st.subheader("Data-quality report")
    frame = report.to_frame()
    st.dataframe(frame if len(frame) else pd.DataFrame({"result": ["no issues detected"]}), width="stretch", hide_index=True)
    if report.actions:
        st.caption("Cleaning actions: " + "; ".join(report.actions))
    if ds is None:
        return
    st.subheader("Sessions")
    c = st.columns(3)
    c[0].metric("Tradable sessions", f"{ds.prep.n_days:,}")
    c[1].metric("First", str(ds.prep.dates[0]) if ds.prep.n_days else "-")
    c[2].metric("Last", str(ds.prep.dates[-1]) if ds.prep.n_days else "-")
    if len(ds.prep.excluded):
        st.write("Excluded sessions by reason:", ds.prep.excluded["reason"].value_counts().to_dict())
        with st.expander("Excluded sessions"):
            st.dataframe(ds.prep.excluded, width="stretch", hide_index=True)
    with st.expander("Per-session completeness (RTH 09:30-16:00 ET)"):
        st.dataframe(report.per_session, width="stretch")
    daily = ds.prep.daily
    fig = plots.go.Figure(plots.go.Scatter(x=daily.index, y=daily["close"] * ds.prep.tick_size, line=dict(color=plots.NET)))
    fig.update_layout(title="Session closes", height=300, **plots.LAYOUT)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Source: {ds.loaded.source} · sha256 {ds.loaded.file_hash[:16]}… · " + "; ".join(ds.loaded.notes))


# ------------------------------------------------------------------------------------------ BACKTEST
def _param_form(defaults: StrategyParams, prefix: str = "bt") -> StrategyParams:
    c = st.columns(4)
    orb_start = c[0].selectbox("ORB start (ET)", TIMES_ORB, index=TIMES_ORB.index(defaults.orb_start), key=f"{prefix}_start")
    rng = c[1].selectbox("Range length (min)", RANGES, index=RANGES.index(defaults.range_minutes), key=f"{prefix}_range")
    entry_method = c[2].selectbox("Entry method", ENTRY_METHODS, index=ENTRY_METHODS.index(defaults.entry_method), key=f"{prefix}_em",
                                  help="market: close-confirmed, fill next bar open · limit: close-confirmed, limit at boundary · stop: intrabar stop order")
    entry_tf = c[3].selectbox("Entry timeframe (min)", ENTRY_TFS, index=ENTRY_TFS.index(defaults.entry_tf), key=f"{prefix}_tf",
                              disabled=entry_method == "stop")
    c = st.columns(4)
    conf = c[0].selectbox("Breakout confirmation", list(CONFIRMATION_PRESETS), key=f"{prefix}_conf")
    buffer = c[1].selectbox("Order buffer (ticks)", [0, 1, 2, 3, 4], key=f"{prefix}_buf", disabled=entry_method == "market")
    stop_choice = c[2].selectbox("Stop", STOP_CHOICES + ["fixed_ticks"], key=f"{prefix}_stop")
    atr_period = c[3].selectbox("ATR period (sessions)", [7, 10, 14, 20], index=2, key=f"{prefix}_atr")
    stop_method, stop_param = (stop_choice.split(":")[0], float(stop_choice.split(":")[1])) if ":" in stop_choice else (stop_choice, 0.0)
    if stop_method == "fixed_ticks":
        stop_param = float(st.number_input("Fixed stop (ticks)", 1, 1000, 8, key=f"{prefix}_fixed"))
    c = st.columns(4)
    target_r = c[0].selectbox("Target (R, 0 = none)", [0.0] + TARGETS, index=TARGETS.index(1.0) + 1, key=f"{prefix}_r")
    cutoff = c[1].selectbox("Entry cutoff (ET)", CUTOFFS, index=CUTOFFS.index(defaults.cutoff), key=f"{prefix}_cut")
    time_exit = c[2].selectbox("Time exit (ET)", TIME_EXITS, key=f"{prefix}_texit")
    direction = c[3].selectbox("Direction", DIRECTIONS, key=f"{prefix}_dir")
    c = st.columns(4)
    max_trades = c[0].selectbox("Max trades/day", [1, 2, 0], format_func=lambda v: "unlimited" if v == 0 else str(v), key=f"{prefix}_mt")
    reentry = c[1].selectbox("Re-entry policy", REENTRY_POLICIES, key=f"{prefix}_re", disabled=max_trades == 1)
    be = c[2].selectbox("Break-even trigger (R, 0 = off)", [0.0, 0.5, 0.75, 1.0, 1.25, 1.5], key=f"{prefix}_be")
    or_filter = c[3].selectbox("OR/ATR filter", ["none", "0.25-0.50", "0.50-0.75", "0.75-1.00", "1.00-1.25", "1.25-1.50"], key=f"{prefix}_orf")
    lo, hi = (0.0, 0.0) if or_filter == "none" else tuple(float(x) for x in or_filter.split("-"))
    ticks, frac = CONFIRMATION_PRESETS[conf]
    return StrategyParams(
        orb_start=orb_start, range_minutes=int(rng), entry_tf=int(entry_tf), entry_method=entry_method, confirm_ticks=ticks,
        confirm_or_frac=frac, entry_buffer_ticks=int(buffer), stop_method=stop_method, stop_param=stop_param, atr_period=int(atr_period),
        target_r=float(target_r), cutoff=cutoff, time_exit=None if time_exit == "session" else time_exit, direction=direction,
        max_trades=int(max_trades), reentry=reentry if max_trades != 1 else "any", breakeven_r=float(be), or_atr_min=lo, or_atr_max=hi,
    )


def _execution_form(inst, prefix="bt") -> ExecutionParams:
    c = st.columns(5)
    slip = c[0].number_input("Slippage (ticks/side)", 0.0, 10.0, float(inst.slippage_ticks), 0.5, key=f"{prefix}_slip")
    comm = c[1].number_input("Commission/side ($)", 0.0, 50.0, float(inst.commission_per_side), 0.05, key=f"{prefix}_comm")
    fee = c[2].number_input("Exchange fee/side ($)", 0.0, 50.0, float(inst.exchange_fee_per_side), 0.05, key=f"{prefix}_fee")
    fill = c[3].selectbox("Limit fill model", FILL_MODELS, index=1, key=f"{prefix}_fill")
    amb = c[4].selectbox("Same-bar ambiguity", AMBIGUITY_MODES, index=2, key=f"{prefix}_amb")
    return ExecutionParams(slippage_ticks=slip, commission_per_side=comm, exchange_fee_per_side=fee, fill_model=fill, ambiguity=amb)


def _date_range(ds, prefix):
    first, last = pd.Timestamp(ds.prep.dates[0]), pd.Timestamp(ds.prep.dates[-1])
    c = st.columns(2)
    start = c[0].date_input("From", first, min_value=first, max_value=last, key=f"{prefix}_from")
    end = c[1].date_input("To", last, min_value=first, max_value=last, key=f"{prefix}_to")
    return start, end


def tab_backtest():
    ds = _state().get("dataset")
    if ds is None:
        st.info("Load data first.")
        return
    inst = _state()["instrument"]
    cfg = load_strategy_config()
    with st.form("backtest_form"):
        st.subheader("Strategy")
        params = _param_form(cfg["strategy"])
        st.subheader("Execution & costs")
        execution = _execution_form(inst)
        st.subheader("Position sizing & period")
        c = st.columns(4)
        mode = c[0].selectbox("Sizing", SIZING_MODES)
        contracts = c[1].number_input("Contracts (fixed)", 1, 1000, 1)
        risk_pct = c[2].number_input("Risk % of equity", 0.1, 10.0, 1.0, 0.1) / 100
        equity = c[3].number_input("Starting equity ($)", 1000.0, 1e9, 100_000.0, 1000.0)
        start, end = _date_range(ds, "bt")
        submitted = st.form_submit_button("RUN BACKTEST", type="primary")
    if submitted:
        sizing = SizingParams(mode=mode, contracts=contracts, risk_dollars=equity * risk_pct, risk_pct=risk_pct, starting_equity=equity)
        try:
            params.validate(ds.prep.base_minutes)
        except ValueError as exc:
            st.error(str(exc))
            return
        with st.spinner("Running..."):
            res = run_backtest(ds.prep, params, execution, sizing, start, end)
            bench = run_backtest(ds.prep, params, replace(execution, frictionless=True), sizing, start, end)
            fixed = run_backtest(ds.prep, params, execution, SizingParams(mode="fixed_contracts", contracts=1, starting_equity=equity), start, end)
        _state()["result"] = res
        _state()["benchmark"] = bench
        _state()["fixed_result"] = fixed
        _state()["backtest_period"] = (start, end)
    res = _state().get("result")
    if res is None:
        return
    bench = _state()["benchmark"]
    m = res.metrics
    st.divider()
    st.caption("All figures are historical backtest results for ONE parameter set chosen by you. They are not evidence of a "
               "persistent edge; use walk-forward results for that.")
    c = st.columns(6)
    c[0].metric("Net P&L", f"${m['net_pnl']:,.0f}")
    c[1].metric("Trades", f"{m['n_trades']:,}")
    c[2].metric("Expectancy (R)", f"{m['avg_r']:.3f}", help=f"t-stat {m['t_stat_r']:.2f}")
    c[3].metric("Profit factor", f"{m['profit_factor']:.2f}")
    c[4].metric("Sharpe", f"{m['sharpe']:.2f}")
    c[5].metric("Max DD", f"{m['max_dd']:.1%}")
    d = res.diagnostics
    if d["ambiguous_exit_pct"] > 0:
        st.warning(f"{d['ambiguous_exits']} trades ({d['ambiguous_exit_pct']:.1%}) exited on bars where stop and target were both "
                   f"inside the bar; resolved with the '{res.execution.ambiguity}' assumption.")
    if d["sized_out_trades"]:
        st.warning(f"{d['sized_out_trades']} signals were skipped because the stop was too wide to size even one contract.")
    table = metrics_table(res.metrics, res.metrics_gross)
    table["Frictionless benchmark (comparison only)"] = metrics_table(bench.metrics)["Net"]
    table["Long only"] = metrics_table(res.metrics_long)["Net"]
    table["Short only"] = metrics_table(res.metrics_short)["Net"]
    fixed = _state().get("fixed_result")
    if fixed is not None and res.sizing.mode != "fixed_contracts":
        table["Fixed 1 contract"] = metrics_table(fixed.metrics)["Net"]
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(plots.equity_and_drawdown(res.daily, bench.daily), width="stretch")
    with right:
        st.dataframe(table, width="stretch", hide_index=True, height=520)
    trades = res.trades
    if len(trades):
        c1, c2 = st.columns(2)
        c1.plotly_chart(plots.r_distribution(trades), width="stretch")
        c2.plotly_chart(plots.pnl_histogram(trades), width="stretch")
        st.plotly_chart(plots.monthly_returns_heatmap(res.daily), width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(plots.annual_returns(res.daily), width="stretch")
        c2.plotly_chart(plots.long_short_equity(trades), width="stretch")
        c1, c2, c3 = st.columns(3)
        wd = trades.assign(weekday=pd.Categorical(trades["weekday"], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], ordered=True))
        c1.plotly_chart(plots.breakdown_bar(wd, "weekday", "Net P&L by weekday"), width="stretch")
        c2.plotly_chart(plots.breakdown_bar(trades, "exit_reason", "Net P&L by exit reason"), width="stretch")
        c3.plotly_chart(plots.breakdown_bar(trades.assign(entry_hour=trades["entry_time"].dt.strftime("%H:00")), "entry_hour",
                                            "Net P&L by entry hour"), width="stretch")
        st.plotly_chart(plots.rolling_metrics(res.daily, trades), width="stretch")
    st.subheader("Transaction-cost sensitivity")
    if st.button("Run slippage sensitivity (0-3 ticks)"):
        start, end = _state()["backtest_period"]
        rows = []
        for slip in SLIPPAGE_GRID:
            r = run_backtest(ds.prep, res.params, replace(res.execution, slippage_ticks=slip), res.sizing, start, end).metrics
            rows.append({"slippage ticks/side": slip, "trades": r["n_trades"], "net P&L": r["net_pnl"], "expectancy R": r["avg_r"],
                         "profit factor": r["profit_factor"], "Sharpe": r["sharpe"], "max DD": r["max_dd"]})
        sens = pd.DataFrame(rows)
        st.dataframe(sens, width="stretch", hide_index=True)
        breakeven = sens[sens["net P&L"] <= 0]["slippage ticks/side"].min()
        if pd.notna(breakeven):
            st.warning(f"Net P&L is not positive from {breakeven} tick(s) of slippage per side.")
    if st.button("Save experiment (manifest + trades + metrics)"):
        path = save_backtest(res, ds, inst)
        st.success(f"Saved to {path}")


# ------------------------------------------------------------------------------------------ OPTIMIZATION
def tab_optimization():
    ds = _state().get("dataset")
    if ds is None:
        st.info("Load data first.")
        return
    inst = _state()["instrument"]
    cfg = load_strategy_config()
    spaces = load_search_spaces(base=cfg["strategy"])
    st.caption("Grid search is IN-SAMPLE. The top row is the 'highest historical result', not an optimal strategy. "
               "Look for broad regions that work, then confirm them with walk-forward testing.")
    preset = st.selectbox("Parameter space", ["custom"] + list(spaces), index=1 + list(spaces).index("primary_930_reduced"))
    if preset != "custom":
        st.caption(spaces[preset].description)
        grid = dict(spaces[preset].grid)
    else:
        grid = {}
    with st.expander("Edit parameter ranges", expanded=preset == "custom"):
        def ms(label, key, options, default):
            chosen = st.multiselect(label, options, default=[v for v in default if v in options] or options[:1], key=f"opt_{key}")
            return chosen
        c = st.columns(3)
        g = {}
        g["orb_start"] = ms("ORB start", "start", TIMES_ORB, grid.get("orb_start", ["09:30"]))
        with c[0]:
            g["range_minutes"] = ms("Range length", "range", RANGES, grid.get("range_minutes", [10, 15, 20]))
            g["entry_tf"] = ms("Entry TF", "tf", ENTRY_TFS, grid.get("entry_tf", [5, 10]))
            g["entry_method"] = ms("Entry method", "em", list(ENTRY_METHODS), grid.get("entry_method", ["market"]))
            g["entry_buffer_ticks"] = ms("Order buffer (ticks)", "buf", [0, 1, 2, 3, 4], grid.get("entry_buffer_ticks", [0]))
        with c[1]:
            g["confirmation"] = ms("Confirmation", "conf", list(CONFIRMATION_PRESETS), grid.get("confirmation", ["close"]))
            g["stop"] = ms("Stop", "stop", STOP_CHOICES, grid.get("stop", ["or_opposite", "or_mid"]))
            g["target_r"] = ms("Target R", "r", TARGETS, grid.get("target_r", [1.0, 1.5]))
        with c[2]:
            g["cutoff"] = ms("Cutoff", "cut", CUTOFFS, grid.get("cutoff", ["11:00"]))
            g["direction"] = ms("Direction", "dir", list(DIRECTIONS), grid.get("direction", ["both"]))
            g["max_trades"] = ms("Max trades/day", "mt", [1, 2, 0], grid.get("max_trades", [1]))
    space = ParameterSpace(preset, {k: v for k, v in g.items() if v}, base=cfg["strategy"])
    expanded = space.expand(ds.prep.base_minutes)
    c = st.columns(4)
    c[0].metric("Raw combinations", f"{expanded.n_raw:,}")
    c[1].metric("Invalid", f"{expanded.n_invalid:,}")
    c[2].metric("Duplicates removed", f"{expanded.n_duplicates:,}")
    c[3].metric("To evaluate", f"{len(expanded.configs):,}")
    with st.form("grid_form"):
        execution = _execution_form(inst, "opt")
        c = st.columns(3)
        workers = c[0].number_input("Worker processes", 1, os.cpu_count() or 1, max(1, (os.cpu_count() or 2) - 1))
        chunk = c[1].number_input("Checkpoint chunk size", 10, 5000, 200)
        resume_dir = c[2].text_input("Resume run directory (optional)", "")
        start, end = _date_range(ds, "opt")
        go_ = st.form_submit_button("RUN OPTIMIZATION", type="primary")
    if go_:
        bar = st.progress(0.0, text="starting...")

        def progress(done, total, elapsed):
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float("nan")
            bar.progress(done / max(total, 1), text=f"{done:,}/{total:,} configs · {elapsed:,.0f}s elapsed · ETA {eta:,.0f}s")

        with st.spinner("Optimizing (checkpointed; safe to resume if interrupted)..."):
            result = run_grid(ds.prep, space, execution, start=start, end=end, n_workers=int(workers), chunk_size=int(chunk),
                              run_dir=resume_dir or None, progress=progress, objective=ObjectiveConfig.from_dict(cfg["objective"]),
                              data_description=ds.describe())
        _state()["grid"] = result
    result = _state().get("grid")
    if result is None or result.results.empty:
        return
    res = result.results
    b = result.selection_bias
    st.subheader("Multiple-testing context")
    msg = (f"{b['n_configs']:,} configurations tested (effective independent trials ≈ {b['n_effective']:,.0f}). "
           f"Best Sharpe {b['best_sharpe']:.2f}; selecting the best of this many *no-edge* configurations would be expected to "
           f"produce ≈ {b['expected_max_sharpe_under_null']:.2f}. {b['share_positive_sharpe']:.0%} of configurations have positive Sharpe.")
    (st.success if b["best_exceeds_null_max"] else st.warning)(msg)
    objective = st.selectbox("Rank by", list(ALTERNATIVE_OBJECTIVES), index=list(ALTERNATIVE_OBJECTIVES).index("composite"),
                             format_func=lambda k: ALTERNATIVE_OBJECTIVES[k])
    ranked = res.sort_values(f"rank_{objective}")
    table = results_table(ranked).assign(Rank=ranked[f"rank_{objective}"].to_numpy())
    st.dataframe(table, width="stretch", hide_index=True, height=420)
    st.download_button("Export results CSV", res.to_csv(index=False).encode(), file_name=f"{result.experiment_id}_results.csv")
    agree = pd.DataFrame({k: res.nsmallest(10, f"rank_{k}")["config_key"].tolist() for k in ALTERNATIVE_OBJECTIVES})
    overlap = {k: len(set(agree[k]) & set(agree["composite"])) for k in ALTERNATIVE_OBJECTIVES}
    st.caption("Overlap of each objective's top-10 with the composite top-10 (low overlap = conclusions depend on the objective): "
               + ", ".join(f"{k}: {v}/10" for k, v in overlap.items()))

    st.subheader("Parameter heatmaps")
    varying = varying_parameters(res)
    if len(varying) < 2:
        st.info("Vary at least two parameters to draw heatmaps.")
        return
    c = st.columns(4)
    pairs = [(x, y) for x, y, _ in STANDARD_PAIRS if x in varying and y in varying]
    default_x, default_y = pairs[0] if pairs else (varying[0], varying[1])
    y = c[0].selectbox("Rows", varying, index=varying.index(default_x))
    x = c[1].selectbox("Columns", varying, index=varying.index(default_y) if default_y != y else 0)
    metric = c[2].selectbox("Metric", ["sharpe", "avg_r", "t_stat_r", "profit_factor", "net_pnl", "max_dd", "is_score", "n_trades", "win_rate"])
    agg = c[3].selectbox("Aggregate hidden parameters by", list(AGGREGATIONS), format_func=lambda k: AGGREGATIONS[k])
    filters = {}
    with st.expander("Fix other parameters (optional)"):
        cols = st.columns(3)
        for i, p in enumerate([v for v in varying if v not in (x, y)]):
            val = cols[i % 3].selectbox(p, ["(all)"] + sorted(res[p].dropna().unique().tolist(), key=str), key=f"hm_fix_{p}")
            if val != "(all)":
                filters[p] = val
    tbl = heatmap_table(res, x, y, metric, agg, filters)
    if tbl.empty:
        st.info("No configurations match the filters.")
    else:
        st.plotly_chart(plots.heatmap(tbl, f"{metric} · {AGGREGATIONS[agg]}", metric, zmid=1.0 if metric == "profit_factor" else 0.0),
                        width="stretch")


# ------------------------------------------------------------------------------------------ later milestones
def tab_pending(name: str, milestone: int, body: str):
    st.info(f"{name} is scheduled for Milestone {milestone} and is not implemented yet. {body}")


def tab_trade_log():
    res = _state().get("result")
    if res is None or res.trades.empty:
        st.info("Run a backtest to see its trade log.")
        return
    trades = res.trades
    c = st.columns(3)
    side = c[0].multiselect("Direction", ["long", "short"], default=["long", "short"])
    reasons = c[1].multiselect("Exit reason", sorted(trades["exit_reason"].unique()), default=sorted(trades["exit_reason"].unique()))
    amb = c[2].checkbox("Only ambiguous exits", value=False)
    view = trades[trades["direction"].isin(side) & trades["exit_reason"].isin(reasons)]
    if amb:
        view = view[view["ambiguous_exit"]]
    cols = ["trade_id", "session_date", "weekday", "direction", "signal_time", "entry_time", "entry_price", "stop_price", "target_price",
            "exit_time", "exit_price", "exit_reason", "qty", "risk_ticks", "gross_pnl", "slippage_cost", "commission", "net_pnl",
            "r_multiple", "mfe_r", "mae_r", "holding_minutes", "or_high", "or_low", "or_width", "atr_daily", "or_atr", "ambiguous_exit",
            "intrabar_entry", "breakeven_moved", "sized_out", "equity_after"]
    st.dataframe(view[[c for c in cols if c in view.columns]], width="stretch", hide_index=True, height=600)
    st.download_button("Export trade log CSV", view.to_csv(index=False).encode(), file_name="trades.csv")
    st.dataframe(res.day_status["status"].value_counts().rename("sessions").to_frame(), width="content")


def tab_report():
    ds = _state().get("dataset")
    res = _state().get("result")
    grid = _state().get("grid")
    if ds is None:
        st.info("Load data first.")
        return
    st.subheader("Data provenance")
    st.json(ds.describe(), expanded=False)
    if res is not None:
        st.subheader("Last backtest (single parameter set; historical)")
        st.json({"strategy": res.params.to_dict(), "execution": res.execution.to_dict(), "sizing": res.sizing.to_dict(),
                 "costs": asdict(res.costs), "diagnostics": res.diagnostics}, expanded=False)
    if grid is not None and not grid.results.empty:
        r = grid.results
        st.subheader("Optimization summary (in-sample)")
        best_hist = r.sort_values("net_pnl", ascending=False).iloc[0]
        best_comp = r.sort_values("rank_composite").iloc[0]
        st.markdown(
            f"- **Highest historical net P&L:** `{best_hist['config_key']}` — {best_hist['range_minutes']}m range, "
            f"{best_hist['entry_tf']}m entry, {best_hist['stop']}, {best_hist['target_r']}R, cutoff {best_hist['cutoff']}, "
            f"{best_hist['direction']}: net ${best_hist['net_pnl']:,.0f}, Sharpe {best_hist['sharpe']:.2f}\n"
            f"- **Best composite in-sample score:** `{best_comp['config_key']}` — Sharpe {best_comp['sharpe']:.2f}, "
            f"t-stat {best_comp['t_stat_r']:.2f}, trades {int(best_comp['n_trades'])}\n"
            f"- Share of configurations with positive net P&L: {(r['net_pnl'] > 0).mean():.1%}\n"
            f"- Run directory: `{grid.run_dir}`"
        )
        st.caption("Best walk-forward family, most robust region, best OOS result and lowest-drawdown viable result require "
                   "Milestones 3-4 and will appear here once implemented.")


def main():
    st.set_page_config(page_title="ORB Lab", layout="wide")
    sidebar()
    tabs = st.tabs(["DATA", "BACKTEST", "OPTIMIZATION", "WALK-FORWARD", "ROBUSTNESS", "MONTE CARLO", "TRADE LOG", "REPORT"])
    with tabs[0]:
        tab_data()
    with tabs[1]:
        tab_backtest()
    with tabs[2]:
        tab_optimization()
    with tabs[3]:
        from .wfo_tab import tab_walk_forward

        tab_walk_forward()
    with tabs[4]:
        tab_pending("Robustness analysis", 4, "It will test neighbouring parameters of every candidate and score parameter plateaus.")
    with tabs[5]:
        tab_pending("Monte Carlo and overfitting diagnostics", 5, "It will add trade reshuffling/bootstrap distributions, Deflated "
                    "Sharpe Ratio and PBO/CSCV. Grid results already report a multiple-testing warning.")
    with tabs[6]:
        tab_trade_log()
    with tabs[7]:
        tab_report()
