"""PHASE-0 RESULTS tab: read-only view of the committed free SPY/QQQ proxy experiment.

Works on a fresh clone without downloading anything: everything shown comes from ``phase0_results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ..data_sources.yahoo import PHASE0_LABEL

RESULTS_DIR = Path(__file__).resolve().parents[2] / "phase0_results"
PAIRS = {
    "ORB duration x R": "range_minutes_x_target_r",
    "ORB duration x stop": "range_minutes_x_stop",
    "R x cutoff": "target_r_x_cutoff",
    "Confirmation x R": "confirmation_x_target_r",
    "ORB duration x entry timeframe": "range_minutes_x_entry_tf",
}


def _load_summary(symbol: str) -> dict | None:
    f = RESULTS_DIR / symbol / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def tab_phase0() -> None:
    st.error(f"**{PHASE0_LABEL}.** SPY and QQQ are ETF proxies. Nothing on this page validates ES, NQ, 6E or GC.")
    symbols = [s for s in ("SPY", "QQQ") if (RESULTS_DIR / s / "summary.json").exists()]
    if not symbols:
        st.info("No Phase-0 results found in `phase0_results/`. Run `python -m orb_lab.cli free-test --symbol SPY,QQQ` to create them.")
        return
    st.markdown(
        "This page shows the completed free experiment: 42 trading days of real Yahoo 5-minute bars (2026-07-29 to 2026-09-25). "
        "Parameters were chosen on the first 33 days, frozen, and then tested once on the last 9 days. "
        "To run your own backtests on the same free data, pick **Free Yahoo (SPY/QQQ)** in the sidebar and press **Load data**."
    )
    cols = st.columns(len(symbols))
    for col, sym in zip(cols, symbols):
        s = _load_summary(sym)
        t = s["headline"]["test"]
        ctrl = s["controls"]["test"]["random_direction"]
        with col:
            st.subheader(sym)
            st.metric("Conclusion", s["verdict"]["category"])
            a, b, c = st.columns(3)
            a.metric("Unseen-test trades", t["trades"])
            b.metric("Expectancy", f"{t['exp_r']:+.2f} R")
            c.metric("Net P&L (100 sh)", f"${t['net_pnl']:,.0f}")
            st.caption(f"Random-direction control p = {ctrl.get('p_value_random_ge_strategy', float('nan')):.2f} "
                       "(values above 0.10 mean the result is not distinguishable from guessing the direction).")

    with st.expander("Full Phase-0 report", expanded=False):
        report = RESULTS_DIR / "PHASE0_REPORT.md"
        if report.exists():
            st.markdown(report.read_text())

    st.divider()
    sym = st.radio("Symbol", symbols, horizontal=True, key="p0_symbol")
    s = _load_summary(sym)
    d = RESULTS_DIR / sym
    sel = s["frozen"]["selected"]
    st.subheader(f"{sym}: the frozen candidate")
    stop = sel["stop_method"] + (f" {sel['stop_param']:g}" if sel["stop_method"] not in ("or_mid", "or_opposite") else "")
    st.write(
        f"Opening range **{sel['range_minutes']} min** from 09:30 · entry **{sel['entry_method']}** "
        f"(entry bars {sel['entry_tf'] or 'intrabar'} min) · stop **{stop}** · target **{sel['target_r']}R** · "
        f"last entry **{sel['cutoff']}** · direction **{sel['direction']}**"
    )
    split = s["split"]
    st.caption(f"Train {split['train'][0]}…{split['train'][1]} · Validation {split['validation'][0]}…{split['validation'][1]} · "
               f"Unseen test {split['final_test'][0]}…{split['final_test'][1]}")
    rows = []
    for ph, label in (("train", "Train"), ("val", "Validation"), ("test", "Unseen test")):
        h = s["headline"][ph]
        rows.append({"phase": label, "trades": h["trades"], "win rate": f"{h['win_rate']:.0%}", "expectancy (R)": round(h["exp_r"], 3),
                     "net P&L (100 sh)": round(h["net_pnl"], 2), "long trades": h["long_trades"], "short trades": h["short_trades"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    tabs = st.tabs(["Trade log", "Trade charts", "Heatmaps", "Costs", "Neighbours"])
    with tabs[0]:
        log_file = d / "trade_log_selected.csv"
        if log_file.exists():
            log = pd.read_csv(log_file)
            st.dataframe(log, hide_index=True, width="stretch", height=420)
            st.download_button("Download trade log (CSV)", log_file.read_bytes(), file_name=f"{sym}_phase0_trade_log.csv")
    with tabs[1]:
        charts = sorted((d / "trade_charts").glob("*.png"))
        kind = st.radio("Show", ["All", "Winners", "Losers", "Selected candidate only", "Verification (other entry types)"], horizontal=True,
                        key="p0_chart_kind")
        if kind == "Winners":
            charts = [c for c in charts if c.stem.endswith("WIN")]
        elif kind == "Losers":
            charts = [c for c in charts if c.stem.endswith("LOSS")]
        elif kind == "Selected candidate only":
            charts = [c for c in charts if not c.name.startswith("verify_")]
        elif kind == "Verification (other entry types)":
            charts = [c for c in charts if c.name.startswith("verify_")]
        st.caption(f"{len(charts)} charts. Orange band = opening range, triangle = entry, dashed lines = stop/target, X = exit.")
        for c in charts:
            st.image(str(c), caption=c.stem, width="stretch")
    with tabs[2]:
        c1, c2 = st.columns(2)
        pair = c1.selectbox("Heatmap", list(PAIRS), key="p0_pair")
        phase = c2.radio("Phase", ["train", "val", "test"], horizontal=True, key="p0_phase",
                         format_func=lambda p: {"train": "Train", "val": "Validation", "test": "Unseen test"}[p])
        img = d / "heatmaps" / f"{PAIRS[pair]}_{phase}.png"
        if img.exists():
            st.image(str(img), width="stretch")
        st.caption("Each cell = median expectancy (R) of all configurations with those two values that traded in that phase. "
                   "Blue = positive, red = negative. Descriptive only.")
    with tabs[3]:
        f = d / "selected_friction_sensitivity.csv"
        if f.exists():
            st.dataframe(pd.read_csv(f)[["split", "friction_usd_per_share", "trades", "exp_r", "net_pnl", "friction_cost"]],
                         hide_index=True, width="stretch")
            st.caption("Friction = $ per share charged on every fill (entry and exit). $0.00 is a reference only.")
    with tabs[4]:
        f = d / "selected_neighbourhood.csv"
        if f.exists():
            st.dataframe(pd.read_csv(f), hide_index=True, width="stretch")
            st.caption("The selected configuration and its one-step neighbours. A real edge should look similar in its neighbours.")
