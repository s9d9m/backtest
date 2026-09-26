"""WALK-FORWARD tab: browse walk-forward experiments produced by ``cli wfo``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..optimization.heatmaps import AGGREGATIONS, heatmap_table
from ..reports import plots
from ..reports.experiment import RUNS_DIR

PHASE_PAIRS = [
    ("range_minutes", "target_r", "Range x R"),
    ("range_minutes", "entry_tf", "Range x Entry TF"),
    ("target_r", "cutoff", "R x Cutoff"),
    ("stop", "target_r", "Stop x R"),
    ("confirmation", "target_r", "Confirmation x R"),
    ("range_minutes", "stop", "Range x Stop"),
    ("orb_start", "range_minutes", "Start time x Range"),
]


def window_timeline(windows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {"train": "#9aa5b1", "val": "#f2b134", "oos": "#2a6fdb"}
    for _, w in windows.iterrows():
        for phase in ("train", "val", "oos"):
            a, b = str(w[f"{phase}_months"]).split("..")
            start = pd.Period(a, "M").start_time
            end = pd.Period(b, "M").end_time
            fig.add_trace(go.Bar(x=[(end - start).total_seconds() * 1000], base=[start], y=[f"W{int(w['window']):02d}"], orientation="h",
                                 marker_color=colors[phase], name=phase, showlegend=bool(w["window"] == 0),
                                 hovertemplate=f"{phase}: {a}..{b}<extra></extra>"))
    fig.update_layout(barmode="overlay", title="Walk-forward windows (grey train, amber validation, blue blind OOS)",
                      height=max(300, 18 * len(windows) + 120), xaxis_type="date", **{k: v for k, v in plots.LAYOUT.items() if k != "hovermode"})
    fig.update_yaxes(autorange="reversed")
    return fig


def tab_walk_forward() -> None:
    runs = sorted(RUNS_DIR.glob("wfo-*"), reverse=True)
    st.caption("Walk-forward runs are produced with `python -m orb_lab.cli wfo ...` (long-running, checkpointed). "
               "Only the stitched blind-OOS segments are evidence; training/validation numbers are shown for comparison.")
    if not runs:
        st.info("No walk-forward runs yet.")
        return
    run = Path(st.selectbox("Experiment", runs, format_func=lambda p: p.name))
    manifest = json.loads((run / "manifest.json").read_text())
    research = "research_mode=True" in manifest.get("notes", [])
    (st.success if research else st.warning)(
        "Research-mode run (DQ-gated, lockbox withheld, registered)." if research else
        "NON-research run (synthetic or ungated data). Not evidence about ORB.")
    overview = pd.read_csv(run / "wfo_overview.csv") if (run / "wfo_overview.csv").exists() else pd.DataFrame()
    st.subheader("All structures and entry families")
    st.dataframe(overview, width="stretch", hide_index=True)
    if overview.empty:
        return
    c = st.columns(2)
    structure = c[0].selectbox("WFO structure", sorted(overview["structure"].unique()))
    family = c[1].selectbox("Entry family", sorted(overview[overview["structure"] == structure]["family"].unique()))
    d = run / structure / family
    summary = json.loads((d / "summary.json").read_text())
    windows = pd.read_csv(d / "windows.csv")
    daily = pd.read_csv(d / "stitched_oos_daily.csv", index_col=0, parse_dates=True)
    flags = summary.get("overfit_flags", [])
    if flags:
        st.error("Overfitting flags: " + "; ".join(flags))
    if summary.get("execution_sensitive"):
        st.error("EXECUTION-SENSITIVE: stitched OOS expectancy turns non-positive within +2 ticks of slippage.")
    s = summary.get("stitched_always_trade", {})
    cols = st.columns(6)
    cols[0].metric("OOS trades", f"{s.get('n_trades', 0):,}")
    cols[1].metric("OOS net P&L (1 ct)", f"${s.get('net_pnl', 0):,.0f}")
    cols[2].metric("OOS mean R", f"{s.get('avg_r', 0):.3f}", help=f"t = {s.get('t_stat_r', 0):.2f}")
    cols[3].metric("OOS profit factor", f"{s.get('profit_factor', 0):.2f}")
    cols[4].metric("OOS Sharpe", f"{s.get('sharpe', 0):.2f}")
    cols[5].metric("Windows profitable", f"{summary.get('pct_windows_oos_profitable', 0):.0%}")
    st.plotly_chart(window_timeline(windows), width="stretch")
    if len(daily):
        st.plotly_chart(plots.equity_and_drawdown(daily, title="Stitched blind-OOS equity (1 contract, net of costs)"), width="stretch")
    comp = pd.DataFrame({
        "phase": ["training (selected)", "validation (selected)", "blind OOS (selected)", "best raw training config: training", "best raw training config: OOS"],
        "median expectancy R": [summary.get("median_train_exp_r"), summary.get("median_val_exp_r"), summary.get("median_oos_exp_r"),
                                summary.get("median_best_train_exp_r"), summary.get("median_best_train_oos_exp_r")]})
    st.subheader("IS vs OOS degradation")
    st.dataframe(comp, hide_index=True)
    st.subheader("Per-window selections (frozen before each OOS window)")
    st.dataframe(windows, width="stretch", hide_index=True)
    st.subheader("Slippage sensitivity of the frozen selections")
    st.dataframe(pd.read_csv(d / "slippage_sensitivity.csv"), hide_index=True)
    st.subheader("Parameter stability (share of windows choosing the modal value)")
    st.json(summary.get("parameter_stability", {}))
    pm_file = run / "phase_metrics.parquet"
    if pm_file.exists():
        st.subheader("Phase heatmaps (descriptive; median across hidden parameters)")
        pm = pd.read_parquet(pm_file)
        c = st.columns(4)
        pairs = [p for p in PHASE_PAIRS if pm[p[0]].nunique() > 1 and pm[p[1]].nunique() > 1]
        if not pairs:
            return
        label = c[0].selectbox("Pair", [p[2] for p in pairs])
        y, x, _ = next(p for p in pairs if p[2] == label)
        metric = c[1].selectbox("Metric", ["exp_r", "sharpe"])
        agg = c[2].selectbox("Aggregation", list(AGGREGATIONS), format_func=lambda k: AGGREGATIONS[k])
        fam = c[3].selectbox("Entry method", ["(all)"] + sorted(pm["entry_method"].unique()))
        filters = {} if fam == "(all)" else {"entry_method": fam}
        cols = st.columns(3)
        for col, phase in zip(cols, ("train", "val", "oos")):
            tbl = heatmap_table(pm, x, y, f"{phase}_{metric}", agg, filters)
            col.plotly_chart(plots.heatmap(tbl, f"{phase.upper()} {metric}", metric), width="stretch")
