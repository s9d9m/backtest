"""OVERVIEW: what is being tested, on what, with which account, how far validation got, and the key numbers."""

from __future__ import annotations

import html

import streamlit as st

from ...reports.final_report import classify
from .. import charts as C
from .. import state as S
from .. import theme as T
from .. import widgets as W
from .results import build_stages


def render() -> None:
    T.page_header("ORB Lab", "Overview", "Opening Range Breakout research: what you are testing, and how far the evidence goes.")
    ds = S.dataset()
    if ds is None:
        _welcome()
        return
    strat = S.strategy()
    inst = S.instrument()
    bt = S.current_backtest()
    if S.is_proxy(inst):
        st.warning(f"**FREE PROXY DATA — NOT FUTURES VALIDATION.** {inst.symbol} ETF bars from Yahoo (about 60 days). "
                   "Useful for learning the workflow; far too short to judge an edge.", icon=":material/info:")
    if S.is_synthetic(ds):
        st.warning("**SYNTHETIC DATA — NOT MARKET EVIDENCE.** For testing the software only.", icon=":material/science:")

    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1:
        T.card("Current strategy", T.chips(S.strategy_chips(), first_is_market=True)
               + f'<div style="margin-top:6px;font-size:0.86rem;color:{T.INK_2}">Signal: {html.escape(S.confirmation_text(strat["params"]))} · '
                 f'source: {html.escape(strat.get("source", ""))}</div>')
    with c2:
        T.card("Portfolio & execution", f'{html.escape(S.portfolio_text())}<div style="margin-top:6px;font-size:0.86rem;color:{T.INK_2}">'
                                        f'{html.escape(S.execution_text())}</div>')
    b = st.columns([1, 1, 1, 1, 2])
    with b[0]:
        S.nav_button("Edit strategy", "strategy", key="ov_edit", icon=":material/edit:", width="stretch")
    with b[1]:
        S.nav_button("Full backtest", "backtest", key="ov_bt", icon=":material/show_chart:", width="stretch")
    with b[2]:
        S.nav_button("Stress test", "stress", key="ov_st", icon=":material/bolt:", width="stretch")
    with b[3]:
        S.nav_button("Validate", "validate", key="ov_val", icon=":material/verified:", width="stretch")

    if bt is None:
        st.error(S.S().get("cur_bt_error", "The strategy could not be backtested on this data."))
        return
    res = bt["res"]
    m = res.metrics
    split = S.S().get("split")
    period = f"{ds.prep.dates[0]} – {ds.prep.dates[-1]}"
    T.section(f"Historical result (in-sample, development data {period})")
    W.headline_cards(m, strat["sizing"].starting_equity)
    W.sized_out_warning(res)
    if split is None:
        T.note("No blind holdout is reserved yet, so these numbers use <b>all</b> loaded sessions. Reserve one on "
               "<b>Validate → Holdout test</b> before optimising, so there is unseen data left to test on.", "warn")
    else:
        T.note(f"The blind holdout ({split['holdout_start']} – {split['holdout_end']}, {split['n_holdout']} sessions) is hidden from "
               "these numbers.", "info")

    T.section("Research stage")
    stages = S.stage_list()
    T.stage_badges(stages, cols=4)
    verdict, reasons = classify(build_stages(for_overview=True))
    T.verdict_banner("Research verdict for this strategy", verdict, reasons[:4])

    if len(res.trades):
        c1, c2 = st.columns([1.6, 1], gap="medium")
        c1.plotly_chart(C.equity(res.daily, "Equity curve (in-sample)", start=strat["sizing"].starting_equity), width="stretch")
        c2.plotly_chart(C.cum_r(res.trades[~res.trades["sized_out"]], "Cumulative R"), width="stretch")

    T.section("Next steps")
    nxt = _next_steps(stages, split)
    cols = st.columns(len(nxt))
    for col, (label, page, why) in zip(cols, nxt):
        with col:
            st.markdown(f"**{label}**")
            st.caption(why)
            S.nav_button(f"Go to {S.PAGES[page][0]}", page, key=f"ov_next_{page}_{label}", width="stretch")


def _next_steps(stages, split) -> list[tuple[str, str, str]]:
    done = {s["name"]: s["tone"] != "none" for s in stages}
    out = []
    if split is None:
        out.append(("Reserve a blind holdout", "validate", "Hide the last part of the data before you optimise."))
    out.append(("Explore settings", "optimize", "Search many settings; look for broad regions, not the single best row."))
    if not done.get("Robustness") or not done.get("Execution") or not done.get("Monte Carlo"):
        out.append(("Stress test", "stress", "Neighbouring settings, worse execution and path risk."))
    if not done.get("Blind OOS") or not done.get("Walk-forward"):
        out.append(("Validate out of sample", "validate", "Only unseen data counts as evidence."))
    out.append(("Read the results", "results", "One page with every stage and the verdict."))
    return out[:4]


def _welcome() -> None:
    T.note("Welcome. ORB Lab tests whether <b>Opening Range Breakout</b> strategies have an edge after realistic costs, using "
           "out-of-sample tests. To begin, load free market data (no account, no card).", "info")
    c = st.columns(3)
    for col, (n, title, text) in zip(c, [("1", "Load data", "Free Yahoo SPY/QQQ 5-minute bars (last ~60 days)."),
                                          ("2", "Define a strategy", "Opening range, entry, stop, target, risk per trade."),
                                          ("3", "Test it honestly", "Backtest, optimise, then validate on unseen data.")]):
        col.markdown(f'<div class="orb-card"><div class="ttl">Step {n}</div><div class="body"><b>{title}</b><br>'
                     f'<span style="color:{T.INK_2};font-size:0.9rem">{text}</span></div></div>', unsafe_allow_html=True)
    W.need_data()
