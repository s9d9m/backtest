"""TRADE EXPLORER: filter trades, inspect one in detail with its chart, export CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .. import charts as C
from .. import logic as L
from .. import state as S
from .. import theme as T
from .. import widgets as W

SOURCES = {"strategy": "Current strategy backtest", "blind": "Blind holdout test", "wfo": "Walk-forward OOS"}
TABLE_COLS = {"trade_id": "#", "session_date": "Date", "direction": "Side", "entry_method": "Entry type", "entry_time": "Entry time",
              "entry_price": "Entry price", "exit_price": "Exit price", "exit_reason": "Exit reason", "qty": "Qty", "risk_dollars": "Risk $",
              "net_pnl": "Net P&L", "r_multiple": "R"}


def _load(source: str) -> tuple[pd.DataFrame | None, str]:
    strat = S.strategy()
    sym = S.instrument().symbol if S.instrument() else ""
    if source == "blind":
        b = S.S().get("blind")
        if not b:
            return None, "Run the blind holdout test (Validate) to see its trades here."
        return b["trades"].assign(market=sym), ""
    if source == "wfo":
        from .validate import _run_dirs

        runs = _run_dirs()
        if not runs:
            return None, "No walk-forward runs yet."
        c1, c2 = st.columns(2)
        run = Path(c1.selectbox("Walk-forward run", [str(r) for r in runs], key="te_run"))
        options = sorted({str(p.parent.relative_to(run)) for p in run.glob("*/*/oos_trades.csv")})
        if not options:
            return None, "This run has no OOS trades."
        sub = c2.selectbox("Structure / family", options, key="te_sub")
        t = pd.read_csv(run / sub / "oos_trades.csv", parse_dates=["session_date", "entry_time", "exit_time", "signal_time"])
        w = pd.read_csv(run / sub / "windows.csv")
        if "param_entry_method" in w:
            t = t.merge(w[["window", "param_entry_method"]].rename(columns={"param_entry_method": "entry_method"}), on="window", how="left")
        return t.assign(market=sym), ""
    bt = S.current_backtest()
    if bt is None:
        return None, "The current strategy has no backtest."
    return bt["res"].trades.assign(entry_method=strat["params"].entry_method, market=sym), ""


def render() -> None:
    T.page_header("Research", "Trade explorer", "Every trade, filterable, with a chart of the session.")
    W.strategy_header()
    if W.need_data():
        return
    source = st.segmented_control("Trades from", list(SOURCES), default="strategy", format_func=SOURCES.get, key="te_source") or "strategy"
    trades, msg = _load(source)
    if trades is None or trades.empty:
        st.info(msg or "No trades.")
        return
    trades = trades.copy()
    trades["session_date"] = pd.to_datetime(trades["session_date"])
    with st.container(border=True):
        c = st.columns([1.3, 1, 1, 1, 1.4])
        d0, d1 = trades["session_date"].min().date(), trades["session_date"].max().date()
        dates = c[0].date_input("Dates", (d0, d1), min_value=d0, max_value=d1, key="te_dates")
        sides = c[1].multiselect("Side", ["long", "short"], default=["long", "short"], key="te_side")
        outcome = c[2].selectbox("Outcome", ["all", "winners", "losers"], key="te_out", format_func=str.capitalize)
        methods = sorted(trades["entry_method"].dropna().unique()) if "entry_method" in trades else []
        pick_m = c[3].multiselect("Entry method", methods, default=methods, key="te_em")
        rmin, rmax = float(trades["r_multiple"].min()), float(trades["r_multiple"].max())
        lo, hi = (min(rmin, -0.01), max(rmax, 0.01))
        r_range = c[4].slider("R result", round(lo - 0.01, 2), round(hi + 0.01, 2), (round(lo - 0.01, 2), round(hi + 0.01, 2)), 0.01, key="te_r")
    start, end = (dates if isinstance(dates, (tuple, list)) and len(dates) == 2 else (d0, d1))
    view = L.filter_trades(trades, start=start, end=end, sides=sides, outcome=outcome, methods=pick_m or None, r_range=r_range)
    if view is None or view.empty:
        st.info("No trades match the filters.")
        return
    wins = (view["net_pnl"] > 0).mean()
    T.kpi_grid([T.kpi("Trades shown", T.count(len(view)), f"of {len(trades)}"),
                T.kpi("Win rate", T.pct(wins, 0), ""), T.kpi("Expectancy", T.rmult(view["r_multiple"].mean()), ""),
                T.kpi("Net P&L", T.usd(view["net_pnl"].sum(), signed=True), "")], min_width=150)
    table = view[[c for c in TABLE_COLS if c in view]].rename(columns=TABLE_COLS)
    pdp = T.price_decimals(S.instrument().tick_size)
    st.dataframe(table, hide_index=True, width="stretch", height=320, column_config={
        "Date": st.column_config.DateColumn(format="YYYY-MM-DD"), "Entry time": st.column_config.DatetimeColumn(format="HH:mm"),
        "Net P&L": st.column_config.NumberColumn(format="dollar"), "Risk $": st.column_config.NumberColumn(format="dollar"),
        "R": st.column_config.NumberColumn(format="%+.2f"),
        "Entry price": st.column_config.NumberColumn(format=f"%.{pdp}f"), "Exit price": st.column_config.NumberColumn(format=f"%.{pdp}f"), "Qty": st.column_config.NumberColumn(format="%.0f")})
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    idx = view.index.tolist()
    labels = {i: f"#{int(view.loc[i, 'trade_id'])} · {view.loc[i, 'session_date'].date()} · {view.loc[i, 'direction']} · "
                 f"{T.rmult(view.loc[i, 'r_multiple'])} · {T.usd(view.loc[i, 'net_pnl'], signed=True)}" for i in idx}
    sel = c1.selectbox("Inspect trade", idx, key="te_pick", format_func=labels.get)
    c2.download_button("Download filtered trades (CSV)", view.to_csv(index=False).encode(), file_name="trades.csv", icon=":material/download:",
                       width="stretch")
    t = view.loc[sel]
    tick = S.instrument().tick_size
    T.kpi_grid([
        T.kpi("Result", T.rmult(t["r_multiple"]), T.usd(t["net_pnl"], signed=True, dp=2), tone="good" if t["net_pnl"] > 0 else "bad"),
        T.kpi("Side & exit", f"{t['direction'].capitalize()}", f"exit: {t['exit_reason']}"),
        T.kpi("Entry", T.price(t["entry_price"], tick),
              f"exit {T.price(t['exit_price'], tick)} · {pd.Timestamp(t['entry_time']).tz_convert('America/New_York'):%H:%M} → "
              f"{pd.Timestamp(t['exit_time']).tz_convert('America/New_York'):%H:%M} ET"),
        T.kpi("Stop", T.price(t["stop_price"], tick), ("target " + (T.price(t["target_price"], tick) if pd.notna(t.get("target_price")) else "none"))
              + f" · risk {T.usd(t.get('risk_dollars'))} · {t.get('qty', 0):,.0f} {S.instrument().unit}s"),
        T.kpi("Costs", T.usd(t.get("total_cost"), dp=2), "commission + fees + friction + slippage"),
    ], min_width=170)
    ds = S.dataset()
    fig = C.trade_chart(ds.prep_full, t, f"{pd.Timestamp(t['session_date']).date()} · orange band = opening range")
    if fig is None:
        st.caption("No price chart available for this trade in the loaded data.")
    else:
        st.plotly_chart(fig, width="stretch")
    with st.expander("Every field of this trade", icon=":material/table:"):
        st.dataframe(t.to_frame("value").astype(str), width="stretch")
