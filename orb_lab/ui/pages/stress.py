"""STRESS TEST: parameter robustness, execution sensitivity and Monte Carlo path risk for the current strategy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ...engine.params import SizingParams
from ...optimization.monte_carlo import METHODS, SIZING, MCConfig, run_monte_carlo, trade_r
from ...optimization.stress import stress_test
from ...optimization.sweeps import AXES, axis_summary, one_at_a_time, overall_verdict, pair_grid
from .. import charts as C
from .. import logic as L
from .. import state as S
from .. import theme as T
from .. import widgets as W

def value_label(axis: str, v: str) -> str:
    """Readable axis values: '50% of range' instead of 'or_pct:0.5'."""
    if axis == "stop":
        return W.stop_choice_label(v).replace("Opposite side of the range", "Opposite side").replace(" of range width", " of range")
    if axis == "entry_method":
        return {"market": "Market", "limit": "Limit", "stop": "Stop order"}.get(v, v)
    if axis == "confirmation":
        return {"close": "Close", "1tick": "+1 tick", "2tick": "+2 ticks", "5pct": "+5% range", "10pct": "+10% range"}.get(v, v)
    if axis == "direction":
        return {"both": "Both", "long": "Long", "short": "Short"}.get(v, v)
    if axis in ("range_minutes", "entry_tf"):
        return f"{v}m"
    if axis == "target_r":
        return f"{float(v):g}R"
    return v


DEFAULT_AXES = ["range_minutes", "entry_tf", "target_r", "stop", "cutoff", "entry_method", "confirmation", "direction"]
CLASS_LABEL = {"plateau": ("Broad plateau", "good"), "mixed": ("Moderate", "warn"), "spike": ("Spike / fragile", "bad"),
               "candidate not positive": ("Not positive", "bad"), "no neighbours": ("No neighbours", "none")}


def render() -> None:
    T.page_header("Research", "Stress test", "Does the result survive small changes to the settings, worse execution, and an unlucky "
                  "order of trades?")
    W.strategy_header()
    if W.need_data():
        return
    t1, t2, t3 = st.tabs(["Parameter robustness", "Execution sensitivity", "Monte Carlo"])
    with t1:
        _robustness()
    with t2:
        _execution()
    with t3:
        _monte_carlo()


def _period():
    ds = S.dataset()
    return pd.Timestamp(ds.prep.dates[0]), pd.Timestamp(ds.prep.dates[-1])


# ---------------------------------------------------------------------------------------------- robustness
def _robustness() -> None:
    st.markdown("If the strategy works because of a real effect, **neighbouring settings should work too** (a plateau). If only the "
                "exact chosen setting works and its neighbours collapse, it is a **spike**: probably a lucky fit to past data.")
    ds = S.dataset()
    strat = S.strategy()
    cand = strat["params"]
    with st.expander("Parameters to vary (one at a time, everything else fixed)", icon=":material/tune:"):
        axes = st.multiselect("Parameters", list(AXES), default=DEFAULT_AXES, format_func=AXES.get, key="rob_axes",
                              label_visibility="collapsed")
    run = st.button("RUN NEIGHBOURHOOD SWEEP", type="primary", key="rob_run", icon=":material/play_arrow:")
    if run:
        bar = st.progress(0.0, text="testing neighbouring settings...")
        table = one_at_a_time(ds.prep, cand, strat["execution"], axes, progress=lambda d, t: bar.progress(d / t, text=f"{d}/{t} settings"))
        summary = axis_summary(table, cand)
        S.store("robustness", {"table": table, "summary": summary, "verdict": overall_verdict(summary)})
        bar.empty()
    rob = S.fetch("robustness")
    if not rob:
        st.caption("Not run for the current strategy yet. It uses the development data (the blind holdout is never touched).")
        return
    label, tone = S.robustness_label(rob["verdict"])
    explain = {"good": "Every tested parameter has neighbours that keep most of the result.",
               "warn": "Some parameters sit on a plateau, others degrade noticeably. Treat the result with caution.",
               "bad": "At least one parameter's neighbours collapse. The chosen setting is likely a lucky fit."}.get(tone, "")
    T.verdict_banner("Parameter robustness", label.upper(), [explain, "In-sample description of the neighbourhood, not new evidence."], tone)
    summ = rob["summary"]
    T.stage_badges([{"name": r.label, "status": CLASS_LABEL.get(r.classification, (r.classification, "none"))[0],
                     "tone": CLASS_LABEL.get(r.classification, ("", "none"))[1],
                     "detail": f"this setting {T.rmult(r.candidate_exp_r)} · neighbours {T.rmult(r.neighbour_median_exp_r)}"}
                    for r in summ.itertuples()])
    table = rob["table"]
    metric = st.segmented_control("Chart metric", ["avg_r", "profit_factor", "n_trades", "sharpe"], default="avg_r", key="rob_metric",
                                  format_func={"avg_r": "Expectancy", "profit_factor": "Profit factor", "n_trades": "Trades",
                                               "sharpe": "Sharpe"}.get) or "avg_r"
    axes_done = list(dict.fromkeys(table["axis"]))
    cols = st.columns(2, gap="medium")
    for i, axis in enumerate(axes_done):
        g = table[table["axis"] == axis].reset_index(drop=True)
        hi = int(np.flatnonzero(g["is_candidate"].to_numpy())[0]) if g["is_candidate"].any() else None
        fmt, suf = {"avg_r": ("+.2f", "R"), "profit_factor": (".2f", ""), "n_trades": (".0f", ""), "sharpe": (".2f", "")}[metric]
        vals = g[metric].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        fig = C.bars([value_label(axis, v) for v in g["value"]], vals, f"{AXES.get(axis, axis)}  (orange = your setting)", fmt=fmt, suffix=suf, highlight=hi, height=280)
        cols[i % 2].plotly_chart(fig, width="stretch")
    st.markdown("**Two parameters at once**")
    c = st.columns([1, 1, 1, 1], vertical_alignment="bottom")
    x = c[0].selectbox("Across", list(AXES), index=list(AXES).index("target_r"), format_func=AXES.get, key="rob_x")
    y = c[1].selectbox("Down", list(AXES), index=list(AXES).index("range_minutes"), format_func=AXES.get, key="rob_y")
    hm_metric = c[2].selectbox("Colour by", ["avg_r", "profit_factor", "sharpe", "n_trades"], key="rob_hm_metric",
                               format_func={"avg_r": "Expectancy (R)", "profit_factor": "Profit factor", "sharpe": "Sharpe", "n_trades": "Trades"}.get)
    if c[3].button("Build heatmap", disabled=x == y, key="rob_hm", width="stretch"):
        bar = st.progress(0.0)
        grid = pair_grid(ds.prep, cand, strat["execution"], x, y, progress=lambda d, t: bar.progress(d / t))
        S.store("rob_pair", (x, y, grid))
        bar.empty()
    pair = S.fetch("rob_pair")
    if pair and len(pair[2]):
        px, py, grid = pair
        tbl = grid.pivot_table(index=py, columns=px, values=hm_metric, aggfunc="first", sort=False)
        cc = grid[grid["is_candidate"]]
        mark = (cc[px].iloc[0], cc[py].iloc[0]) if len(cc) else None
        fmt = {"avg_r": "+.2f", "n_trades": ".0f"}.get(hm_metric, ".2f")
        st.plotly_chart(C.heatmap(tbl, f"{AXES[py]} × {AXES[px]} (orange square = your setting)", fmt=fmt,
                                  zmid=1.0 if hm_metric == "profit_factor" else 0.0, mark=mark), width="stretch")
    with st.expander("Advanced: every evaluated setting", icon=":material/table:"):
        st.dataframe(table, hide_index=True, width="stretch")
        st.dataframe(summ, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------------------------- execution
def _execution() -> None:
    st.markdown("Re-runs the current strategy with **worse execution** than assumed: more slippage, higher costs, pessimistic handling "
                "of bars that hit both stop and target, and entries a tick or two worse. A strategy that only works with perfect fills "
                "is not tradable.")
    ds = S.dataset()
    strat = S.strategy()
    if st.button("RUN EXECUTION STRESS TEST", type="primary", key="stress_run"):
        with st.spinner("Re-simulating under worse execution..."):
            S.store("stress", stress_test(ds.prep, strat["params"], strat["execution"],
                                          SizingParams(mode="fixed_contracts", contracts=1, starting_equity=strat["sizing"].starting_equity)))
    res = S.fetch("stress")
    if not res:
        st.caption("Not run for the current strategy yet.")
        return
    table, info = res
    v = info["verdict"]
    tone = {"ROBUST": "good", "FRAGILE": "bad", "NOT POSITIVE": "bad"}.get(v, "none")
    reasons = {"ROBUST": ["Expectancy stays positive under every moderate stress (+1 tick, 2× costs, pessimistic bars, 1 tick worse entry)."],
               "FRAGILE": ["Positive with your assumptions, but not under: " + ", ".join(info["breaks_under"]) + "."],
               "NOT POSITIVE": ["Expectancy is not positive even with your baseline assumptions."],
               "NO TRADES": ["No trades in the period."]}[v]
    T.verdict_banner("Execution sensitivity", v, reasons, tone)
    base = table.iloc[0]
    worst = table.iloc[-1]
    T.kpi_grid([T.kpi("Baseline expectancy", T.rmult(base["expectancy_r"]), "your assumptions", tone="good" if base["expectancy_r"] > 0 else "bad"),
                T.kpi("With +1 tick slippage", T.rmult(table[table["scenario"] == "slippage +1 tick/side"]["expectancy_r"].iloc[0])
                      if (table["scenario"] == "slippage +1 tick/side").any() else "—", "per side"),
                T.kpi("Combined worst case", T.rmult(worst["expectancy_r"]), "+1 tick, 2× costs, pessimistic",
                      tone="good" if worst["expectancy_r"] > 0 else "bad"),
                T.kpi("Cost per trade (baseline)", T.usd(base["cost_per_trade"], dp=2), f"1 {info['unit']}")], min_width=180)
    st.plotly_chart(C.bars(table["scenario"], table["expectancy_r"], "Expectancy under each scenario (orange = baseline)", highlight=0,
                           horizontal=True, height=90 + 34 * len(table)), width="stretch")
    st.caption("'Adverse entry' is an approximation of a delayed fill (entry k ticks worse, same exits); every other row is a full re-simulation.")
    with st.expander("Advanced: scenario table", icon=":material/table:"):
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "expectancy_r": st.column_config.NumberColumn("Expectancy R", format="%+.3f"), "net_pnl": st.column_config.NumberColumn(format="dollar"),
            "profit_factor": st.column_config.NumberColumn(format="%.2f"), "win_rate": st.column_config.NumberColumn(format="percent"),
            "cost_per_trade": st.column_config.NumberColumn(format="dollar")})


# ---------------------------------------------------------------------------------------------- Monte Carlo
SOURCES = {"strategy": "Current strategy (development data)", "blind": "Blind holdout trades", "wfo": "Walk-forward OOS trades"}


def _mc_trades(source: str):
    if source == "blind":
        b = S.S().get("blind")
        return (b["trades"] if b else None), "blind holdout trades"
    if source == "wfo":
        from .validate import _run_dirs

        runs = _run_dirs()
        if not runs:
            return None, "no walk-forward runs"
        run = Path(st.selectbox("Walk-forward run", [str(r) for r in runs], key="mc_wfo_run"))
        options = sorted({str(p.parent.relative_to(run)) for p in run.glob("*/*/oos_trades.csv")})
        if not options:
            return None, "this run has no OOS trades"
        sub = st.selectbox("Structure / family", options, key="mc_wfo_sub")
        return pd.read_csv(run / sub / "oos_trades.csv", parse_dates=["entry_time"]), f"stitched OOS trades ({sub})"
    bt = S.current_backtest()
    return (bt["res"].trades if bt else None), "current strategy on development data (in-sample)"


def _monte_carlo() -> None:
    st.markdown("Monte Carlo **re-orders or re-samples the historical trades** thousands of times to show how different the balance, "
                "drawdown and losing streaks could have been. It describes **path risk**; it cannot tell you whether an in-sample edge is real.")
    strat = S.strategy()
    sz = strat["sizing"]
    source = st.segmented_control("Trades to simulate", list(SOURCES), default="strategy", format_func=SOURCES.get, key="mc_source") or "strategy"
    trades, label = _mc_trades(source)
    r = trade_r(trades) if trades is not None else np.zeros(0)
    if len(r) < 2:
        st.info(f"Not enough executed trades to simulate ({label}).")
        if source == "strategy" and trades is not None and len(trades) and trades["sized_out"].any():
            W.sized_out_warning(S.current_backtest()["res"])
        return
    if source == "strategy":
        st.warning("These trades are in-sample if the strategy was chosen on this data. Simulating them shows path risk only.",
                   icon=":material/info:")
    with st.form("mc_form", border=True):
        c = st.columns(4)
        method = c[0].selectbox("Resampling method", list(METHODS), format_func=METHODS.get, index=1)
        n_sims = c[1].select_slider("Simulations", [1000, 2000, 5000, 10000, 20000], value=5000)
        equity = c[2].number_input("Starting balance ($)", 1000.0, 1e9, float(sz.starting_equity), 1000.0, format="%.0f")
        risk_default = sz.risk_pct * 100 if sz.mode != "fixed_contracts" else 1.0
        risk = c[3].number_input("Risk per trade (%)", 0.05, 10.0, float(round(risk_default, 3)), 0.25)
        with st.expander("More options"):
            c = st.columns(5)
            sizing = c[0].selectbox("Sizing", list(SIZING), format_func=SIZING.get, index=1 if sz.mode == "pct_equity" else 0)
            seed = c[1].number_input("Random seed", 0, 2**31 - 1, 12345, help="Same seed = identical result.")
            block = c[2].number_input("Block size (block bootstrap)", 1, 100, 5)
            skip = c[3].slider("Skip probability (missed trades)", 0.0, 0.5, 0.1, 0.05)
            extra = c[4].number_input("Extra cost per trade (R)", 0.0, 1.0, 0.0, 0.01)
        go_ = st.form_submit_button("RUN MONTE CARLO", type="primary", icon=":material/casino:")
    if go_:
        cfg = MCConfig(method=method, n_sims=int(n_sims), seed=int(seed), starting_equity=float(equity), risk_pct=float(risk) / 100,
                       sizing=sizing, block_size=int(block), skip_prob=float(skip), extra_cost_r=float(extra))
        with st.spinner("Simulating..."):
            out = {"result": run_monte_carlo(r, cfg), "label": label, "source": source}
        if source == "strategy":
            S.store("mc", out)
        else:
            S.S()["mc_other"] = out
    out = S.fetch("mc") if source == "strategy" else S.S().get("mc_other")
    if not out or out.get("source") != source:
        return
    res = out["result"]
    h = L.mc_headline(res)
    cfg = res.config
    st.caption(f"{out['label']} · {res.n_trades} trades · {cfg.n_sims:,} simulations · {METHODS[cfg.method]} · "
               f"${cfg.starting_equity:,.0f} at {cfg.risk_pct * 100:g}% risk/trade ({SIZING[cfg.sizing].lower()}) · seed {cfg.seed}")
    T.kpi_grid([
        T.kpi("Historical ending balance", T.usd(h["hist_end"]), f"{T.pct(h['hist_return'], signed=True)} · what actually happened", big=True,
              tone="good" if h["hist_return"] > 0 else "bad"),
        T.kpi("Median simulated ending balance", T.usd(h["median_end"]), f"{T.pct(h['median_end'] / h['start'] - 1, signed=True)}", big=True),
        T.kpi("Probability of loss", T.pct(h["p_loss"], 0), "simulations ending below the start", big=True,
              tone="good" if h["p_loss"] < 0.1 else "warn" if h["p_loss"] < 0.3 else "bad"),
        T.kpi("Bad case ending balance", T.usd(h["p5_end"]), "5th percentile", big=True, tone="bad" if h["p5_end"] < h["start"] else "none"),
    ], min_width=210)
    T.kpi_grid([
        T.kpi("Historical max drawdown", T.pct(h["hist_dd"]), "the path that happened"),
        T.kpi("Median simulated max drawdown", T.pct(h["median_dd"]), "typical path"),
        T.kpi("Bad-case drawdown", T.pct(h["p5_dd"]), "5th percentile (1 in 20 paths is worse)", tone="bad" if h["p5_dd"] < -0.2 else "warn"),
        T.kpi("Losing streak", f"{h['median_streak']:.0f} trades", f"typical · {h['p95_streak']:.0f} in a bad case"),
    ], min_width=210)
    c1, c2 = st.columns(2, gap="medium")
    c1.plotly_chart(C.hist(res.sims["ending_equity"], h["hist_end"], "Ending balance", xfmt="$,.0f"), width="stretch")
    c2.plotly_chart(C.hist(res.sims["max_drawdown"], h["hist_dd"], "Maximum drawdown", xfmt=".0%"), width="stretch")
    c1, c2 = st.columns(2, gap="medium")
    c1.plotly_chart(C.hist(res.sims["longest_losing_streak"], res.historical["longest_losing_streak"], "Longest losing streak (trades)",
                           discrete=True), width="stretch")
    c2.plotly_chart(C.hist(res.sims["total_return"], h["hist_return"], "Total return", xfmt=".0%"), width="stretch")
    st.caption("Grey = simulations, orange line = historical, dark line = median simulation.")
    st.plotly_chart(C.sim_paths(res.sample_paths, res.historical_path), width="stretch")
    with st.expander("Advanced simulation statistics", icon=":material/table:"):
        tbl = res.percentiles.copy()
        tbl.loc["historical"] = pd.Series(res.historical)
        st.dataframe(tbl, width="stretch", column_config={
            "ending_equity": st.column_config.NumberColumn(format="dollar"), "total_return": st.column_config.NumberColumn(format="percent"),
            "max_drawdown": st.column_config.NumberColumn(format="percent"), "max_drawdown_r": st.column_config.NumberColumn(format="%.2f R"),
            "longest_losing_streak": st.column_config.NumberColumn(format="%.0f"), "sum_r": st.column_config.NumberColumn(format="%+.2f R")})
        st.dataframe(pd.Series(res.probabilities, name="probability").to_frame(), width="content",
                     column_config={"probability": st.column_config.NumberColumn(format="percent")})
        if cfg.method == "reshuffle" and cfg.sizing == "fixed_risk":
            st.caption("Reshuffling with fixed $ risk keeps the same trades, so every path ends at the same balance; only drawdowns and streaks vary.")
