"""MONTE CARLO tab: resample a trade sequence to see the range of paths it could have produced."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..engine.backtest import run_backtest
from ..engine.params import SizingParams
from ..optimization.monte_carlo import METHODS, SIZING, MCConfig, run_monte_carlo, trade_r
from ..reports import plots
from . import common
from .wfo_tab import _run_dirs

SOURCES = ["Candidate (backtest on development data)", "Last BACKTEST tab result", "Walk-forward stitched OOS trades",
           "Blind holdout test trades"]


def _hist(values: np.ndarray, hist_value: float, title: str, fmt: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, nbinsx=60, marker_color="#9aa5b1", name="simulated"))
    fig.add_vline(x=hist_value, line_color="#ff7f0e", line_width=3, annotation_text="historical", annotation_position="top")
    fig.update_layout(title=title, height=300, showlegend=False, **{k: v for k, v in plots.LAYOUT.items() if k != "hovermode"})
    fig.update_xaxes(tickformat=fmt)
    return fig


def _trades_for(source: str) -> tuple[pd.DataFrame | None, str]:
    s = common.state()
    ds = common.dataset()
    if source == SOURCES[1]:
        res = s.get("result")
        return (res.trades if res is not None else None), "last backtest"
    if source == SOURCES[2]:
        runs = _run_dirs()
        if not runs:
            return None, "no walk-forward runs"
        run = st.selectbox("Walk-forward run", [str(r) for r in runs], key="mc_wfo_run")
        from pathlib import Path

        options = sorted({str(p.parent.relative_to(Path(run))) for p in Path(run).glob("*/*/oos_trades.csv")})
        if not options:
            return None, "run has no OOS trades"
        sub = st.selectbox("Structure / family", options, key="mc_wfo_sub")
        t = pd.read_csv(Path(run) / sub / "oos_trades.csv", parse_dates=["entry_time"])
        return t, f"stitched OOS {sub}"
    if source == SOURCES[3]:
        blind = s.get("blind")
        return (blind["trades"] if blind else None), "blind holdout"
    pick = common.candidate_picker("Candidate", key="mc_cand")
    if pick is None or ds is None:
        return None, "no candidate"
    name, cand = pick
    key = ("mc_cand_trades", cand.key())
    if s.get("mc_cache_key") != key:
        res = run_backtest(ds.prep, cand, s.get("robustness", {}).get("execution") if s.get("robustness") else None,
                           SizingParams(mode="fixed_contracts", contracts=1))
        s["mc_cache_key"], s["mc_cache"] = key, res.trades
    return s["mc_cache"], f"candidate {name} on development data (in-sample if it was optimised here)"


def tab_monte_carlo() -> None:
    st.markdown("Monte Carlo **re-orders or re-samples the historical trade outcomes** (in R, net of costs) thousands of times to "
                "show how different the equity path, drawdown and losing streaks could have been. It adds no new market information: "
                "if the trades have no edge, resampling them will not create one.")
    source = st.radio("Trades to simulate", SOURCES, horizontal=True, key="mc_source")
    trades, label = _trades_for(source)
    r = trade_r(trades) if trades is not None else np.zeros(0)
    if len(r) < 2:
        st.info(f"Not enough trades to simulate ({label}). Run a backtest, a walk-forward or pick a candidate first.")
        return
    if source != SOURCES[2] and source != SOURCES[3]:
        st.warning("These trades are in-sample if the configuration was chosen on the same data. Simulating them describes path risk, "
                   "not whether the edge is real. Prefer walk-forward OOS or blind holdout trades.")
    st.caption(f"Source: {label} · {len(r)} trades · historical mean {r.mean():+.3f} R")
    with st.form("mc_form"):
        c = st.columns(4)
        method = c[0].selectbox("Resampling method", list(METHODS), format_func=METHODS.get, index=1)
        n_sims = c[1].number_input("Number of simulations", 100, 100_000, 5000, 500)
        seed = c[2].number_input("Random seed (same seed = same result)", 0, 2**31 - 1, 12345)
        sizing = c[3].selectbox("Sizing", list(SIZING), format_func=SIZING.get)
        c = st.columns(5)
        equity = c[0].number_input("Starting equity ($)", 1000.0, 1e9, 40_000.0, 1000.0)
        risk = c[1].selectbox("Risk per trade", [0.25, 0.5, 1.0, 2.0], index=2, format_func=lambda v: f"{v} %")
        block = c[2].number_input("Block size (block bootstrap)", 1, 100, 5)
        skip = c[3].slider("Skip probability (missed trades)", 0.0, 0.5, 0.1, 0.05)
        extra = c[4].number_input("Extra cost per trade (R)", 0.0, 1.0, 0.0, 0.01, help="Cost stress: subtract this many R from every trade.")
        go_ = st.form_submit_button("RUN MONTE CARLO", type="primary")
    if go_:
        cfg = MCConfig(method=method, n_sims=int(n_sims), seed=int(seed), starting_equity=float(equity), risk_pct=float(risk) / 100,
                       sizing=sizing, block_size=int(block), skip_prob=float(skip), extra_cost_r=float(extra))
        with st.spinner("Simulating..."):
            common.state()["mc"] = {"result": run_monte_carlo(r, cfg), "label": label}
    out = common.state().get("mc")
    if not out:
        return
    res = out["result"]
    h, p = res.historical, res.probabilities
    st.subheader("Historical vs simulated")
    st.caption(f"{out['label']} · {res.n_trades} trades · {res.config.n_sims:,} simulations · {METHODS[res.config.method]} · seed "
               f"{res.config.seed}. Orange lines / 'historical' = what actually happened; grey = simulated.")
    c = st.columns(4)
    c[0].metric("Probability of ending with a loss", f"{p['p_loss']:.0%}")
    c[1].metric("Historical max drawdown", f"{h['max_drawdown']:.1%}",
                help=f"{p['p_dd_worse_than_historical']:.0%} of simulated paths had a worse drawdown")
    c[2].metric("Median simulated max drawdown", f"{res.percentiles.loc['50th', 'max_drawdown']:.1%}")
    c[3].metric("5th percentile drawdown (bad case)", f"{res.percentiles.loc['5th', 'max_drawdown']:.1%}")
    tbl = res.percentiles.copy()
    tbl.loc["historical"] = pd.Series(h)
    st.dataframe(tbl.style.format({"ending_equity": "${:,.0f}", "total_return": "{:.1%}", "max_drawdown": "{:.1%}", "max_drawdown_r": "{:.2f} R",
                                   "longest_losing_streak": "{:.0f}", "sum_r": "{:+.2f} R"}), width="stretch")
    st.dataframe(pd.Series(p, name="probability").to_frame().style.format("{:.1%}"), width="content")
    c1, c2 = st.columns(2)
    c1.plotly_chart(_hist(res.sims["ending_equity"], h["ending_equity"], "Ending equity ($)", ",.0f"), width="stretch")
    c2.plotly_chart(_hist(res.sims["max_drawdown"], h["max_drawdown"], "Maximum drawdown", ".0%"), width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(_hist(res.sims["total_return"], h["total_return"], "Total return", ".0%"), width="stretch")
    c2.plotly_chart(_hist(res.sims["longest_losing_streak"], h["longest_losing_streak"], "Longest losing streak (trades)", "d"), width="stretch")
    fig = go.Figure()
    for path in res.sample_paths:
        fig.add_trace(go.Scatter(y=path, mode="lines", line=dict(color="rgba(150,160,170,0.25)", width=1), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(y=res.historical_path, mode="lines", line=dict(color="#ff7f0e", width=3), name="historical"))
    fig.update_layout(title=f"{len(res.sample_paths)} simulated equity paths vs historical", height=420, xaxis_title="trade #",
                      yaxis_title="equity ($)", **{k: v for k, v in plots.LAYOUT.items() if k != "hovermode"})
    st.plotly_chart(fig, width="stretch")
    if res.config.method == "reshuffle" and res.config.sizing == "fixed_risk":
        st.caption("Reshuffling with fixed $ risk keeps the same trades, so every path ends at the same equity; only drawdowns and streaks vary.")
