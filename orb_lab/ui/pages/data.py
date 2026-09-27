"""DATA: sources, loading, data quality, sessions and provenance. Prepared (inactive) panel for Databento futures."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from ...engine.data_quality import CleaningPolicy
from ...engine.pipeline import DataQualityError
from .. import charts as C
from .. import state as S
from .. import theme as T

SOURCES = [S.YAHOO, "File path", "Upload", "Synthetic (testing only)"]
FUTURES = [("ES", "E-mini S&P 500"), ("NQ", "E-mini Nasdaq-100"), ("GC", "COMEX Gold"), ("6E", "Euro FX")]


def render() -> None:
    T.page_header("Data", "Market data", "Where prices come from, whether they are clean, and exactly what was loaded.")
    ds, inst = S.dataset(), S.instrument()
    if ds is not None:
        rep = ds.report.summary()
        T.kpi_grid([
            T.kpi("Market", inst.symbol, inst.name.split(" (")[0]),
            T.kpi("Sessions", T.count(ds.prep_full.n_days), f"{ds.prep_full.dates[0]} – {ds.prep_full.dates[-1]}"),
            T.kpi("Bar interval", f"{rep['bar_minutes']} min", f"{rep['rows_clean']:,} bars"),
            T.kpi("Data quality", "Pass" if ds.report.ok and rep["n_errors"] == 0 else "Errors", f"{rep['n_errors']} errors · {rep['n_warnings']} warnings",
                  tone="good" if rep["n_errors"] == 0 else "bad"),
            T.kpi("Kind", "ETF proxy" if S.is_proxy(inst) else "Synthetic" if S.is_synthetic(ds) else "Futures",
                  "not futures validation" if S.is_proxy(inst) else "free Yahoo continuous" if S.is_free_futures(ds, inst) else "",
                  tone="warn" if S.is_proxy(inst) or S.is_synthetic(ds) or S.is_free_futures(ds, inst) else "none"),
        ], min_width=170)
    _loader()
    ds = S.dataset()
    blocked = S.S().get("dq_block")
    if ds is None and blocked is None:
        _databento()
        return
    report = ds.report if ds is not None else blocked
    T.section("Data quality")
    if not report.ok:
        st.error("Unresolved data-quality errors block research. Choose an explicit cleaning policy below, or fix the file.")
    frame = report.to_frame()
    if len(frame):
        st.dataframe(frame, width="stretch", hide_index=True)
    else:
        T.note("No data-quality issues detected.", "good")
    if report.actions:
        st.caption("Cleaning actions: " + "; ".join(report.actions))
    if ds is not None:
        daily = ds.prep_full.daily
        fig = go.Figure(go.Scatter(x=daily.index, y=daily["close"] * ds.prep_full.tick_size, line=dict(color=T.BLUE, width=2)))
        st.plotly_chart(C.style(fig, "Session closes", 280), width="stretch")
        c1, c2 = st.columns(2)
        with c1.expander(f"Excluded sessions ({len(ds.prep_full.excluded)})", icon=":material/event_busy:"):
            if len(ds.prep_full.excluded):
                st.dataframe(ds.prep_full.excluded, width="stretch", hide_index=True)
            else:
                st.caption("None.")
        with c2.expander("Per-session completeness (09:30–16:00 ET)", icon=":material/fact_check:"):
            st.dataframe(report.per_session, width="stretch")
        with st.expander("Provenance (source, file hash, cleaning policy)", icon=":material/fingerprint:"):
            st.json(ds.describe(), expanded=False)
    _databento()


def _loader() -> None:
    etfs = S.etf_instruments()
    free = S.free_instruments()
    with st.container(border=True):
        st.markdown("**Load data**")
        c = st.columns([1.6, 1])
        source = c[0].radio("Source", SOURCES, horizontal=True, key="data_source")
        refresh, symbol, path, upload, tz, convention = False, None, None, None, None, "start"
        s_start, s_end, seed, trend = "2021-01-04", "2021-12-30", 7, 0.0
        if source == S.YAHOO:
            cc = st.columns([1, 2])
            symbol = cc[0].selectbox("Market", list(free), key="data_etf",
                                     format_func=lambda k: f"{k} · Euro FX futures (contracts)" if k == "6E" else f"{k} · ETF (shares)")
            inst = free[symbol]
            saved = S.yahoo_saved_copy(symbol)
            cc[1].caption(("Cached copy found (" + str(saved.name) + ")." if saved else "No cached copy: the last ~60 days will be downloaded (about 30 s).")
                          + (" Free, no account. Yahoo 6E=F: continuous front-month futures, 5-minute bars, full Globex hours; "
                             "sized in contracts ($6.25 per 0.00005 tick)." if symbol == "6E" else
                             " Free, no account. FREE PROXY — NOT FUTURES VALIDATION."))
            refresh = cc[1].checkbox("Download fresh data from Yahoo", value=False, key="data_refresh")
        else:
            instruments = S.all_instruments()
            cc = st.columns([1, 2])
            sym = cc[0].selectbox("Market specification", list(instruments), key="data_market",
                                  format_func=lambda k: f"{k} (ETF, shares)" if k in etfs else f"{k} (futures, contracts)")
            inst = instruments[sym]
            cc[1].caption(inst.cost_description().replace("$", "\\$"))
            if source == "File path":
                default = next(iter(sorted(Path("data").glob("*.parquet"))), None) if Path("data").exists() else None
                path = st.text_input("Path to CSV or Parquet", value=str(default) if default else "", key="data_path")
            elif source == "Upload":
                upload = st.file_uploader("CSV or Parquet (timestamp, open, high, low, close, volume)", type=["csv", "parquet", "pq", "txt"])
            else:
                cc = st.columns(4)
                s_start = cc[0].text_input("Start", "2021-01-04", key="data_s_start")
                s_end = cc[1].text_input("End", "2021-12-30", key="data_s_end")
                seed = cc[2].number_input("Seed", value=7, step=1, key="data_seed")
                trend = cc[3].slider("Planted trend (0 = no edge)", 0.0, 2.0, 0.0, 0.1, key="data_trend")
            if source in ("File path", "Upload"):
                cc = st.columns(2)
                tz_ = cc[0].selectbox("Timezone of timestamps without offset", ["(timestamps carry an offset)", "America/New_York",
                                                                                 "America/Chicago", "UTC"], key="data_tz")
                tz = None if tz_.startswith("(") else tz_
                convention = cc[1].selectbox("Timestamps mark the", ["start", "end"], format_func=lambda x: f"bar {x}", key="data_conv")
        with st.expander("Cleaning policy and session rules", icon=":material/cleaning_services:"):
            cc = st.columns(5)
            dup = cc[0].selectbox("Conflicting duplicates", ["error", "keep_first", "keep_last"], key="data_dup")
            inv = cc[1].selectbox("Invalid OHLC rows", ["error", "drop"], key="data_inv")
            allow = cc[2].checkbox("Allow unresolved errors (recorded)", value=False, key="data_allow")
            coverage = cc[3].slider("Min opening-range coverage", 0.5, 1.0, 0.8, 0.05, key="data_cov")
            excl = cc[4].checkbox("Exclude early-close sessions", value=False, key="data_excl")
        if st.button("Load data", type="primary", key="data_load", icon=":material/download:"):
            kind = {SOURCES[0]: S.YAHOO, SOURCES[3]: "Synthetic"}.get(source, "File")
            with st.spinner("Loading and checking data..."):
                try:
                    msg = S.load_dataset(kind, inst, symbol=symbol, refresh=refresh, path=path, upload=upload, tz=tz, convention=convention,
                                         start=s_start, end=s_end, seed=int(seed), trend=float(trend), policy=CleaningPolicy(dup, inv),
                                         allow_errors=allow, min_or_coverage=coverage, exclude_early_close=excl)
                    st.success(f"Loaded {inst.symbol}: {msg}.")
                except DataQualityError:
                    st.error("Data-quality errors block research. See the report below.")
                except Exception as exc:
                    st.error(str(exc))


def _databento() -> None:
    T.section("Real futures data (prepared — not active)")
    st.caption("The futures pipeline is built and tested with mocked downloads. No paid data has been requested and no API key is "
               "configured. Nothing on this page can spend money.")
    cols = st.columns(4)
    for col, (sym, name) in zip(cols, FUTURES):
        col.markdown(f'<div class="orb-card"><div class="ttl">{sym} · CME Globex</div><div class="body"><b>{name}</b><br>'
                     f'<span style="color:{T.INK_2};font-size:0.86rem">GLBX.MDP3 · OHLCV-1m · 2015–2026 · individual contracts, '
                     f'causal front-month roll (no back-adjustment)</span><br><span class="orb-chip grey">not downloaded</span></div></div>',
                     unsafe_allow_html=True)
    with st.expander("How the futures data will be added", icon=":material/help:"):
        st.markdown(
            "1. A cost estimate is free: `python -m orb_lab.cli fetch --instrument ES --start 2015-06-01 --end 2026-09-01 --estimate-only`.\n"
            "2. The download needs a Databento key and runs with a hard budget cap (`--budget`).\n"
            "3. `dq-report` checks 09:30 ET alignment month by month, DST transitions, holidays and early closes, and seals a 12-month "
            "**lockbox** that no page can see until a one-shot final test.\n"
            "4. The file then loads here with *File path*, and every page works unchanged. Micro contracts (MES, MNQ, MGC, M6E) "
            "can reuse the same prices for smaller accounts.\n\n"
            "These steps use the terminal on purpose, because they involve a paid key. See DATA_SOURCES.md.")
