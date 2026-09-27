"""ROBUSTNESS tab: parameter-neighbourhood sweeps around a candidate (plateau vs spike)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..optimization.sweeps import AXES, METRICS, axis_summary, one_at_a_time, overall_verdict, pair_grid
from ..reports import plots
from . import common

DEFAULT_AXES = ["range_minutes", "entry_tf", "target_r", "stop", "cutoff", "entry_method", "confirmation", "direction"]
METRIC_LABELS = {"avg_r": "Expectancy (R)", "sharpe": "Sharpe", "t_stat_r": "t-stat", "profit_factor": "Profit factor",
                 "net_pnl": "Net P&L (1 unit)", "n_trades": "Trades", "win_rate": "Win rate", "max_dd": "Max drawdown"}


def _axis_chart(table: pd.DataFrame, axis: str, metric: str) -> go.Figure:
    g = table[table["axis"] == axis]
    colors = ["#ff7f0e" if c else ("#2a6fdb" if v > 0 else "#d62728") for c, v in zip(g["is_candidate"], g[metric])]
    fig = go.Figure(go.Bar(x=g["value"], y=g[metric], marker_color=colors, text=g["n_trades"].astype(int).astype(str) + " tr",
                           textposition="outside"))
    fig.update_layout(title=f"{AXES.get(axis, axis)} (orange = candidate)", height=300, yaxis_title=METRIC_LABELS.get(metric, metric),
                      **{k: v for k, v in plots.LAYOUT.items() if k != "hovermode"})
    fig.update_xaxes(type="category")
    return fig


def tab_robustness() -> None:
    ds = common.dataset()
    if ds is None:
        st.info("Load data first (left sidebar → Load data).")
        return
    st.markdown("Robustness asks: **if the parameters were slightly different, would the result survive?** A real effect usually "
                "degrades gradually around the chosen setting (a *plateau*). If neighbouring settings collapse, the candidate is "
                "probably a lucky *spike* found by the optimizer.")
    st.caption("These sweeps are in-sample descriptions of the neighbourhood on the period you pick. They are not new evidence.")
    common.dev_notice()
    pick = common.candidate_picker("Candidate", key="rob_cand")
    if pick is None:
        st.info("No candidate yet. In BACKTEST, OPTIMIZATION or WALK-FORWARD press **Use as candidate**.")
        return
    name, cand = pick
    st.caption(f"Candidate: {common.describe_params(cand)}")
    inst = common.instrument()
    with st.form("rob_form"):
        axes = st.multiselect("Parameters to vary (one at a time; everything else fixed at the candidate)", list(AXES),
                              default=DEFAULT_AXES, format_func=AXES.get)
        with st.expander("Costs used for every evaluation", expanded=False):
            execution = common.execution_form(inst, "rob")
        first, last = pd.Timestamp(ds.prep.dates[0]), pd.Timestamp(ds.prep.dates[-1])
        c = st.columns(2)
        start = c[0].date_input("From", first, min_value=first, max_value=last, key="rob_from")
        end = c[1].date_input("To", last, min_value=first, max_value=last, key="rob_to")
        go_ = st.form_submit_button("RUN NEIGHBOURHOOD SWEEP", type="primary")
    if go_:
        bar = st.progress(0.0, text="sweeping...")
        table = one_at_a_time(ds.prep, cand, execution, axes, start, end,
                              progress=lambda d, t: bar.progress(d / t, text=f"{d}/{t} configurations"))
        summary = axis_summary(table, cand)
        common.state()["robustness"] = {"candidate": name, "table": table, "summary": summary, "verdict": overall_verdict(summary),
                                        "execution": execution, "period": (start, end)}
    rob = common.state().get("robustness")
    if not rob or rob["candidate"] != name:
        return
    verdict = rob["verdict"]
    (st.success if verdict.startswith("PLATEAU") else st.error if verdict.startswith(("SPIKE", "NOT")) else st.warning)(f"Verdict: **{verdict}**")
    st.dataframe(rob["summary"].style.format({"candidate_exp_r": "{:+.3f}", "neighbour_median_exp_r": "{:+.3f}",
                                               "neighbour_min_exp_r": "{:+.3f}", "share_of_range_positive": "{:.0%}"}),
                 hide_index=True, width="stretch")
    st.caption("plateau: every immediate neighbour is positive and their median keeps ≥ 50 % of the candidate's expectancy. "
               "spike: the neighbours' median keeps ≤ 25 % or is not positive. Share of range positive = how much of the whole axis works.")
    metric = st.selectbox("Metric for charts", list(METRIC_LABELS), format_func=METRIC_LABELS.get, key="rob_metric")
    table = rob["table"]
    axes_done = list(dict.fromkeys(table["axis"]))
    cols = st.columns(2)
    for i, axis in enumerate(axes_done):
        cols[i % 2].plotly_chart(_axis_chart(table, axis, metric), width="stretch")
    with st.expander("Full table (degradation vs the candidate)"):
        st.dataframe(table, hide_index=True, width="stretch")
    st.subheader("Two-parameter heatmap around the candidate")
    c = st.columns(3)
    x = c[0].selectbox("Columns", list(AXES), index=list(AXES).index("target_r"), format_func=AXES.get, key="rob_x")
    y = c[1].selectbox("Rows", list(AXES), index=list(AXES).index("range_minutes"), format_func=AXES.get, key="rob_y")
    hm_metric = c[2].selectbox("Metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get, key="rob_hm_metric")
    if st.button("Build heatmap", disabled=x == y):
        bar = st.progress(0.0)
        grid = pair_grid(ds.prep, cand, rob["execution"], x, y, *rob["period"], progress=lambda d, t: bar.progress(d / t))
        common.state()["rob_pair"] = (x, y, grid)
    pair = common.state().get("rob_pair")
    if pair and len(pair[2]):
        px, py, grid = pair
        tbl = grid.pivot_table(index=py, columns=px, values=hm_metric, aggfunc="first", sort=False)
        cand_cell = grid[grid["is_candidate"]]
        title = f"{METRIC_LABELS.get(hm_metric, hm_metric)}; candidate at {py}={cand_cell[py].iloc[0]}, {px}={cand_cell[px].iloc[0]}" \
            if len(cand_cell) else METRIC_LABELS.get(hm_metric, hm_metric)
        st.plotly_chart(plots.heatmap(tbl, title, hm_metric, zmid=1.0 if hm_metric == "profit_factor" else 0.0), width="stretch")
