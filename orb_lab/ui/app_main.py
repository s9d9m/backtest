"""ORB Lab dashboard shell: page setup, sidebar navigation and routing.

Navigation (research first, data and administration last):
Overview · Strategy · Backtest · Optimize · Validate · Stress test · Results · Trade explorer · Data · Settings & research
Only the selected page is rendered, so each page stays fast.
"""

from __future__ import annotations

import html

import streamlit as st

from .. import __version__
from . import state as S
from . import theme as T
from .pages import backtest, data, optimize, overview, results, settings, strategy, stress, trades, validate

RENDER = {"overview": overview.render, "strategy": strategy.render, "backtest": backtest.render, "optimize": optimize.render,
          "validate": validate.render, "stress": stress.render, "results": results.render, "trades": trades.render, "data": data.render,
          "settings": settings.render}


def sidebar() -> str:
    with st.sidebar:
        st.markdown(f'<div style="font-size:1.45rem;font-weight:800;letter-spacing:-0.02em;margin:-8px 0 0 2px">ORB Lab</div>'
                    f'<div style="color:{T.MUTED};font-size:0.8rem;margin:0 0 14px 2px">Opening Range Breakout research · v{__version__}</div>',
                    unsafe_allow_html=True)
        if "nav" not in st.session_state:
            st.session_state["nav"] = "overview"
        page = st.radio("Navigation", list(S.PAGES), key="nav", label_visibility="collapsed",
                        format_func=lambda k: f"{S.PAGES[k][1]} {S.PAGES[k][0]}")
        st.divider()
        ds, inst = S.dataset(), S.instrument()
        if ds is None:
            st.caption("No data loaded")
        else:
            kind = ("free ETF proxy" if S.is_proxy(inst) else "synthetic" if S.is_synthetic(ds) else
                    "free Yahoo futures" if S.is_free_futures(ds, inst) else "futures")
            split = st.session_state.get("split")
            st.markdown(f'<div style="font-size:0.84rem;color:{T.INK_2};line-height:1.55"><b>{html.escape(inst.symbol)}</b> · {kind}<br>'
                        f'{ds.prep_full.n_days} sessions · {ds.prep_full.dates[0]} – {ds.prep_full.dates[-1]}<br>'
                        f'{"blind holdout reserved" if split else "no holdout reserved"}</div>', unsafe_allow_html=True)
    return page


def main() -> None:
    st.set_page_config(page_title="ORB Lab", page_icon=":material/candlestick_chart:", layout="wide", initial_sidebar_state="expanded")
    T.inject_css()
    S.autoload()
    S.strategy()
    page = sidebar()
    RENDER.get(page, overview.render)()
