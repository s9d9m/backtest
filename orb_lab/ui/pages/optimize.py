"""OPTIMIZE: search many settings, and separate the best historical row from a robust candidate."""

from __future__ import annotations

import html
import os

import pandas as pd
import streamlit as st

from ...engine.params import DIRECTIONS, ENTRY_METHODS, load_strategy_config
from ...optimization.grid_search import results_table, run_grid
from ...optimization.heatmaps import AGGREGATIONS, STANDARD_PAIRS, heatmap_table, varying_parameters
from ...optimization.objectives import ALTERNATIVE_OBJECTIVES, ObjectiveConfig
from ...optimization.parameter_space import CONFIRMATION_PRESETS, ParameterSpace, load_search_spaces, params_from_row
from ...optimization.robustness import grid_stability
from .. import charts as C
from .. import logic as L
from .. import state as S
from .. import theme as T
from .. import widgets as W

LARGE_SEARCH = 5_000
DD_BASE = 100_000.0  # grid metrics are fractions of this notional equity at 1 unit (grid_search.evaluate_config default)
METRIC_LABELS = {"avg_r": "Expectancy (R)", "sharpe": "Sharpe", "t_stat_r": "t-stat", "profit_factor": "Profit factor", "net_pnl": "Net P&L",
                 "n_trades": "Trades", "win_rate": "Win rate", "max_dd": "Max drawdown", "is_score": "Composite score"}


def render() -> None:
    T.page_header("Research", "Optimize", "Test many combinations of settings at once. The purpose is to find broad regions that work — "
                  "not the single best row, which is mostly luck.")
    W.strategy_header()
    if W.need_data():
        return
    ds = S.dataset()
    split = S.S().get("split")
    if split is None:
        st.warning("**No blind holdout is reserved.** Anything you optimise now sees all the data, leaving nothing unseen to test "
                   "on. Reserve a holdout first on Validate → Holdout test.", icon=":material/lock_open:")
        S.nav_button("Reserve a blind holdout", "validate", key="opt_to_split", icon=":material/lock:")
    _search_form(ds, split)
    result = S.S().get("grid")
    if result is None or result.results.empty:
        return
    _results(result)


def _search_form(ds, split) -> None:
    cfg = load_strategy_config()
    spaces = load_search_spaces(base=cfg["strategy"])
    reduced = [k for k in spaces if "reduced" in k or k == "smoke"]
    base = ds.prep.base_minutes
    with st.container(border=True):
        st.markdown("**What to search**")
        c1, c2 = st.columns([2, 3])
        preset = c1.selectbox("Search preset", list(spaces), index=list(spaces).index("primary_930_reduced"), key="opt_preset",
                              format_func=lambda k: f"{k} · {'reduced' if k in reduced else 'comprehensive'}")
        c2.caption(spaces[preset].description)
        grid = dict(spaces[preset].grid)
        with st.expander("Edit the ranges searched", icon=":material/tune:"):
            def ms(label, key, options, default):
                return st.multiselect(label, options, default=[v for v in default if v in options] or options[:1], key=f"opt_{key}")
            c = st.columns(3)
            g = {"orb_start": grid.get("orb_start", ["09:30"])}
            with c[0]:
                g["range_minutes"] = ms("Range length", "range", [r for r in W.RANGES if r % base == 0], grid.get("range_minutes", [10, 15, 20]))
                g["entry_tf"] = ms("Entry TF", "tf", [t for t in W.ENTRY_TFS if t % base == 0], grid.get("entry_tf", [5, 10]))
                g["entry_method"] = ms("Entry method", "em", list(ENTRY_METHODS), grid.get("entry_method", ["market"]))
                g["entry_buffer_ticks"] = ms("Order buffer (ticks)", "buf", [0, 1, 2, 3, 4], grid.get("entry_buffer_ticks", [0]))
            with c[1]:
                g["confirmation"] = ms("Confirmation", "conf", list(CONFIRMATION_PRESETS), grid.get("confirmation", ["close"]))
                g["stop"] = ms("Stop", "stop", [s for s in W.STOPS if s != "fixed_ticks"] + ["or_pct:1.0", "atr:1.0", "atr:2.0"],
                               grid.get("stop", ["or_opposite", "or_mid"]))
                g["target_r"] = ms("Target R", "r", [t for t in W.TARGETS if t > 0], grid.get("target_r", [1.0, 1.5]))
            with c[2]:
                g["cutoff"] = ms("Cutoff", "cut", W.CUTOFFS, grid.get("cutoff", ["11:00"]))
                g["direction"] = ms("Direction", "dir", list(DIRECTIONS), grid.get("direction", ["both"]))
                g["max_trades"] = ms("Max trades/day", "mt", [1, 2, 0], grid.get("max_trades", [1]))
        space = ParameterSpace(preset, {k: v for k, v in g.items() if v}, base=cfg["strategy"])
        expanded = space.expand(base)
        n_eval = len(expanded.configs)
        T.kpi_grid([T.kpi("Combinations", T.count(expanded.n_raw), "before cleaning"),
                    T.kpi("Invalid / duplicate", T.count(expanded.n_invalid + expanded.n_duplicates), "removed"),
                    T.kpi("Configurations to test", T.count(n_eval), "reduced" if n_eval <= LARGE_SEARCH else "comprehensive",
                          tone="warn" if n_eval > LARGE_SEARCH else "none")], min_width=160)
        first, last = pd.Timestamp(ds.prep.dates[0]), pd.Timestamp(ds.prep.dates[-1])
        end_default = min(pd.Timestamp(split["train_end"]), last) if split else last
        with st.form("grid_form", border=False):
            c = st.columns([1, 1, 1, 1])
            start = c[0].date_input("From", first, min_value=first, max_value=last, key="opt_from")
            end = c[1].date_input("To", end_default, min_value=first, max_value=last, key="opt_to",
                                  help="With a holdout reserved this defaults to the TRAIN period, keeping validation unseen.")
            workers = c[2].number_input("Worker processes", 1, os.cpu_count() or 1, max(1, (os.cpu_count() or 2) - 1))
            resume_dir = c[3].text_input("Resume folder (optional)", "", help="Continue an interrupted run from its checkpoint folder.")
            confirm = True
            if n_eval > LARGE_SEARCH:
                confirm = st.checkbox(f"I understand this is a comprehensive search of {n_eval:,} configurations: it can take a long time "
                                      "and the best row will be even more inflated by luck.", value=False)
            st.caption("Costs and fills come from the current strategy's execution assumptions. Each configuration is evaluated with "
                       "1 unit, so compounding cannot distort the comparison.")
            go_ = st.form_submit_button("RUN OPTIMIZATION", type="primary", icon=":material/play_arrow:")
    if go_ and not confirm:
        st.error("Tick the confirmation box to run a comprehensive search, or choose a reduced preset.")
        return
    if go_:
        bar = st.progress(0.0, text="starting...")

        def progress(done, total, elapsed):
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float("nan")
            left = f"about {eta:,.0f}s left" if eta == eta else "estimating time left"
            bar.progress(done / max(total, 1), text=f"{done:,}/{total:,} configurations · {elapsed:,.0f}s · {left}")

        execution = S.strategy()["execution"]
        result = run_grid(ds.prep, space, execution, start=start, end=end, n_workers=int(workers), chunk_size=200,
                          run_dir=resume_dir or None, progress=progress, objective=ObjectiveConfig.from_dict(cfg["objective"]),
                          data_description=ds.describe())
        S.S()["grid"] = result
        S.S()["grid_period"] = (str(start), str(end))
        S.S()["grid_execution"] = execution
        S.S().pop("grid_stability", None)
        bar.empty()


def _card(title: str, row: pd.Series | None, why: str, tone: str) -> None:
    if row is None:
        T.card(title, f'<span style="color:{T.INK_2}">{html.escape(why)}</span>')
        return
    p = params_from_row(row.to_dict())
    stab = L.STABILITY_LABEL.get(row.get("stability"), row.get("stability", ""))
    body = T.chips(S.strategy_chips(p, S.instrument().symbol), first_is_market=True)
    st.markdown(f'<div class="orb-card" style="border-top:4px solid {T.TONE_COLOR[tone]}"><div class="ttl">{html.escape(title)}</div>'
                f'<div class="body">{body}<div style="margin-top:8px;font-size:0.9rem;color:{T.INK_2}">'
                f'<b>{T.rmult(row["avg_r"])}</b> expectancy · PF <b>{T.ratio(row["profit_factor"])}</b> · <b>{int(row["n_trades"])}</b> trades · '
                f'Sharpe {T.ratio(row["sharpe"])} · max DD (1 unit) {T.usd(row["max_dd"] * DD_BASE)} · t {T.ratio(row["t_stat_r"])} · '
                f'top-5 share {T.pct(row["top5_share"], 0)} · stability <b>{html.escape(stab)}</b></div>'
                f'<div style="margin-top:6px;font-size:0.84rem;color:{T.MUTED}">{html.escape(why)}</div></div></div>', unsafe_allow_html=True)


def _actions(row: pd.Series, source: str, key: str) -> None:
    p = params_from_row(row.to_dict())
    st.button("Use this strategy", key=f"{key}_use", type="primary", on_click=S.use_strategy, args=(p, source), width="stretch",
              icon=":material/check:")
    c = st.columns(3)
    c[0].button("Backtest", key=f"{key}_bt", on_click=S.use_strategy, args=(p, source, "backtest"), width="stretch")
    c[1].button("Validate", key=f"{key}_val", on_click=S.use_strategy, args=(p, source, "validate"), width="stretch")
    c[2].button("Stress test", key=f"{key}_st", on_click=S.use_strategy, args=(p, source, "stress"), width="stretch")


def _results(result) -> None:
    res = result.results
    b = result.selection_bias
    stab = S.S().get("grid_stability")
    if stab is None or len(stab) != len(res):
        stab = grid_stability(res, result.expanded.configs, "avg_r")
        S.S()["grid_stability"] = stab
    full = res.join(stab)
    period = S.S().get("grid_period", ("?", "?"))

    T.section(f"Results · in-sample {period[0]} – {period[1]}")
    lucky = not b["best_exceeds_null_max"]
    (st.warning if lucky else st.info)(
        f"**Multiple testing:** {b['n_configs']:,} configurations were tested (≈ {b['n_effective']:,.0f} independent tries). "
        f"Picking the best of that many settings **with no edge at all** would still show a Sharpe of about "
        f"{b['expected_max_sharpe_under_null']:.2f}; the best here is {b['best_sharpe']:.2f}. "
        + ("The best result does NOT beat what luck alone would produce." if lucky else
           "The best result exceeds that, but it is still in-sample and must be validated.")
        + f" {b['share_positive_sharpe']:.0%} of configurations have positive Sharpe.", icon=":material/casino:")

    best = L.best_historical(full)
    robust, why = L.robust_candidate(full)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _card("Best historical configuration", best, "Highest in-sample net P&L. Usually the luckiest fit, not the best strategy.", "warn")
        if best is not None:
            _actions(best, "optimizer: best historical (in-sample)", "best")
    with c2:
        _card("Robust candidate", robust, why, "good" if robust is not None and robust["stability"] == "plateau" else "warn")
        if robust is not None:
            _actions(robust, "optimizer: robust candidate (in-sample)", "robust")
    if best is not None and robust is not None and best["config_key"] == robust["config_key"]:
        st.caption("Here the best historical configuration is also the most robust candidate.")

    T.section("Top configurations")
    c1, c2 = st.columns([1, 3])
    objective = c1.selectbox("Rank by", list(ALTERNATIVE_OBJECTIVES), index=list(ALTERNATIVE_OBJECTIVES).index("composite"),
                             format_func=lambda k: ALTERNATIVE_OBJECTIVES[k], key="opt_rank")
    ranked = full.sort_values(f"rank_{objective}").head(50)
    view = pd.DataFrame({
        "Rank": ranked[f"rank_{objective}"].astype(int).to_numpy(),
        "Strategy": [S.params_text(params_from_row(r.to_dict())) for _, r in ranked.iterrows()],
        "Expectancy R": ranked["avg_r"].to_numpy(), "PF": ranked["profit_factor"].to_numpy(), "Trades": ranked["n_trades"].astype(int).to_numpy(),
        "Sharpe": ranked["sharpe"].to_numpy(), "Max DD (1 unit)": ranked["max_dd"].to_numpy() * DD_BASE, "t-stat": ranked["t_stat_r"].to_numpy(),
        "Top-5 share": ranked["top5_share"].to_numpy(), "Stability": [L.STABILITY_LABEL.get(s, s) for s in ranked["stability"]],
    })
    st.dataframe(view, hide_index=True, width="stretch", height=380, column_config={
        "Expectancy R": st.column_config.NumberColumn(format="%+.3f", help="Average net R per trade"),
        "PF": st.column_config.NumberColumn(format="%.2f", help="Profit factor"), "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Max DD (1 unit)": st.column_config.NumberColumn(format="dollar", help="Approximate largest $ drawdown trading 1 contract/share (the "
                                                                                     "optimizer compares configurations at fixed size)"),
        "t-stat": st.column_config.NumberColumn(format="%.2f"),
        "Top-5 share": st.column_config.NumberColumn(format="percent", help="Share of profit from the 5 best trades. Above 100% means all "
                                                                           "the other trades together lost money."),
        "Stability": st.column_config.TextColumn(width="medium", help="Broad plateau = neighbouring settings also work; spike = they collapse"),
        "Strategy": st.column_config.TextColumn(width="large")})
    labels = {i: f"#{int(ranked.loc[i, f'rank_{objective}'])} · {T.rmult(ranked.loc[i, 'avg_r'])} · {int(ranked.loc[i, 'n_trades'])} trades · "
                 f"{L.STABILITY_LABEL.get(ranked.loc[i, 'stability'], '')} · {S.params_text(params_from_row(ranked.loc[i].to_dict()))}"
              for i in ranked.index}
    pick = c2.selectbox("Act on a configuration", ranked.index.tolist(), key="opt_pick", format_func=labels.get)
    with c2:
        _actions(ranked.loc[pick], f"optimizer rank {int(ranked.loc[pick, f'rank_{objective}'])} ({objective}, in-sample)", "pick")

    with st.expander("Heatmaps: which regions work?", icon=":material/grid_on:"):
        _heatmaps(res)
    with st.expander(f"Advanced: all {len(res):,} rows and every column", icon=":material/table:"):
        table = results_table(full.sort_values(f"rank_{objective}")).assign(Rank=full.sort_values(f"rank_{objective}")[f"rank_{objective}"].to_numpy())
        st.dataframe(table, width="stretch", hide_index=True, height=420)
        st.download_button("Download all results (CSV)", res.to_csv(index=False).encode(), file_name=f"{result.experiment_id}_results.csv")
        agree = pd.DataFrame({k: res.nsmallest(10, f"rank_{k}")["config_key"].tolist() for k in ALTERNATIVE_OBJECTIVES})
        overlap = {k: len(set(agree[k]) & set(agree["composite"])) for k in ALTERNATIVE_OBJECTIVES}
        st.caption("Overlap of each objective's top 10 with the composite top 10 (low overlap = conclusions depend on the objective): "
                   + ", ".join(f"{k}: {v}/10" for k, v in overlap.items()) + f". Run folder: {result.run_dir}")


def _heatmaps(res: pd.DataFrame) -> None:
    varying = varying_parameters(res)
    if len(varying) < 2:
        st.info("Vary at least two parameters to draw heatmaps.")
        return
    c = st.columns(4)
    pairs = [(x, y) for x, y, _ in STANDARD_PAIRS if x in varying and y in varying]
    dx, dy = pairs[0] if pairs else (varying[0], varying[1])
    y = c[0].selectbox("Rows", varying, index=varying.index(dx), key="hm_rows")
    x = c[1].selectbox("Columns", varying, index=varying.index(dy) if dy != y else 0, key="hm_cols")
    metric = c[2].selectbox("Colour by", list(METRIC_LABELS), format_func=METRIC_LABELS.get, key="hm_metric")
    agg = c[3].selectbox("Combine hidden settings by", list(AGGREGATIONS), format_func=lambda k: AGGREGATIONS[k], key="hm_agg")
    tbl = heatmap_table(res, x, y, metric, agg, {})
    if tbl.empty:
        st.info("No configurations match.")
        return
    fmt = {"avg_r": "+.2f", "net_pnl": ",.0f", "n_trades": ".0f", "win_rate": ".0%", "max_dd": ".0%"}.get(metric, ".2f")
    st.plotly_chart(C.heatmap(tbl, f"{METRIC_LABELS[metric]} · {AGGREGATIONS[agg]}", fmt=fmt, zmid=1.0 if metric == "profit_factor" else 0.0),
                    width="stretch")
    st.caption("Blue = better, red = worse. A robust edge shows up as a broad blue region, not a single bright cell.")
