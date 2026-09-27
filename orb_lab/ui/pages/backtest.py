"""BACKTEST: did the current strategy make or lose money on the development data, and how?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...reports import plots
from ...reports.experiment import metrics_table, save_backtest
from .. import charts as C
from .. import state as S
from .. import theme as T
from .. import widgets as W


def render() -> None:
    T.page_header("Research", "Backtest", "Historical performance of the current strategy. In-sample: this is where ideas start, "
                  "not where they are proven.")
    W.strategy_header()
    if W.need_data():
        return
    ds = S.dataset()
    strat = S.strategy()
    first, last = pd.Timestamp(ds.prep.dates[0]), pd.Timestamp(ds.prep.dates[-1])
    c1, c2, c3 = st.columns([1, 1, 2], vertical_alignment="bottom")
    start = c1.date_input("From", first, min_value=first, max_value=last, key="bt_from")
    end = c2.date_input("To", last, min_value=first, max_value=last, key="bt_to")
    split = S.S().get("split")
    c3.caption(("Development data only: the blind holdout "
                f"({split['holdout_start']} – {split['holdout_end']}) is hidden." if split else
                "All loaded sessions (no blind holdout reserved yet — see Validate).") + " Results update automatically.")
    bt = S.current_backtest(None if pd.Timestamp(start) == first else start, None if pd.Timestamp(end) == last else end)
    if bt is None:
        st.error(S.S().get("cur_bt_error", "This strategy cannot run on the loaded data."))
        return
    res, bench = bt["res"], bt["bench"]
    m = res.metrics
    unit = res.diagnostics.get("unit", "contract")
    trades = res.trades[~res.trades["sized_out"]] if len(res.trades) else res.trades

    W.headline_cards(m, strat["sizing"].starting_equity)
    W.metrics_explained(m)
    d = res.diagnostics
    W.sized_out_warning(res)
    if d.get("size_capped_trades"):
        st.info(f"{d['size_capped_trades']} trades were reduced by the position-value cap (see Strategy → Advanced).")
    if d["ambiguous_exit_pct"] > 0.05:
        st.warning(f"{d['ambiguous_exit_pct']:.0%} of exits happened on bars containing both stop and target; resolved with the "
                   f"'{res.execution.ambiguity}' assumption.")
    if not len(trades):
        st.info("No trades in this period.")
        return

    c1, c2 = st.columns(2, gap="medium")
    c1.plotly_chart(C.equity(res.daily, start=strat["sizing"].starting_equity, gross=True), width="stretch")
    c2.plotly_chart(C.drawdown(res.daily), width="stretch")
    c1, c2 = st.columns(2, gap="medium")
    c1.plotly_chart(C.cum_r(trades), width="stretch")
    c2.plotly_chart(C.trade_pnl(trades), width="stretch")

    T.section("Details")
    long_m, short_m = res.metrics_long, res.metrics_short
    T.kpi_grid([
        T.kpi("Long trades", T.count(long_m["n_trades"]), f"{T.rmult(long_m['avg_r'])} · {T.usd(long_m['net_pnl'], signed=True)}"),
        T.kpi("Short trades", T.count(short_m["n_trades"]), f"{T.rmult(short_m['avg_r'])} · {T.usd(short_m['net_pnl'], signed=True)}"),
        T.kpi("Cost per trade", T.usd(m["cost_per_trade"], dp=2), f"{T.usd(m['total_cost'])} in total"),
        T.kpi("Top-5 share", T.pct(m["top5_share"], 0) if m["net_pnl"] > 0 else "n/a", f"P&L without top 5: {T.usd(m['pnl_ex_top5'], signed=True)}",
              tone="warn" if (m["top5_share"] or 0) > 0.5 else "none"),
        T.kpi("Sharpe", T.ratio(m["sharpe"]), "annualised" + (" · < 1 year!" if m["years"] < 1 else ""), tone="warn" if m["years"] < 1 else "none"),
        T.kpi("Sortino", T.ratio(m["sortino"]), "annualised"),
        T.kpi("t-stat", T.ratio(m["t_stat_r"]), "≥ 2 needed, and more after searching", tone="warn" if abs(m["t_stat_r"]) < 2 else "none"),
        T.kpi("Longest losing streak", T.count(m["longest_loss_streak"]), "consecutive losing trades"),
    ], min_width=150)
    c1, c2 = st.columns(2, gap="medium")
    c1.plotly_chart(C.position_size(trades, unit), width="stretch")
    by_reason = trades.groupby("exit_reason")["net_pnl"].sum()
    c2.plotly_chart(C.bars(by_reason.index, by_reason.values, "Net P&L by exit reason", fmt=",.0f", suffix="", text=[T.usd(v) for v in by_reason.values], yfmt="$,.0f"),
                    width="stretch")

    with st.expander("More charts: monthly returns, weekdays, entry hour, rolling statistics", icon=":material/insights:"):
        st.plotly_chart(plots.monthly_returns_heatmap(res.daily), width="stretch")
        c1, c2 = st.columns(2)
        wd = trades.groupby("weekday")["net_pnl"].sum().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]).fillna(0)
        c1.plotly_chart(C.bars(wd.index, wd.values, "Net P&L by weekday", fmt=",.0f", suffix="", text=[T.usd(v) for v in wd.values], yfmt="$,.0f"), width="stretch")
        hr = trades.assign(h=trades["entry_time"].dt.tz_convert("America/New_York").dt.strftime("%H:00")).groupby("h")["net_pnl"].sum()
        c2.plotly_chart(C.bars(hr.index, hr.values, "Net P&L by entry hour (ET)", fmt=",.0f", suffix="", text=[T.usd(v) for v in hr.values], yfmt="$,.0f"),
                        width="stretch")
        st.plotly_chart(plots.rolling_metrics(res.daily, trades), width="stretch")
    with st.expander("Advanced: full statistics (net, gross, frictionless benchmark, long, short)", icon=":material/table:"):
        table = metrics_table(res.metrics, res.metrics_gross)
        table["Frictionless benchmark (comparison only)"] = metrics_table(bench.metrics)["Net"]
        table["Long only"] = metrics_table(res.metrics_long)["Net"]
        table["Short only"] = metrics_table(res.metrics_short)["Net"]
        st.dataframe(table, width="stretch", hide_index=True, height=560)
        st.caption("Trade-by-trade details are in the Trade Explorer.")
    c1, c2, _ = st.columns([1.3, 1, 1.5])
    with c1:
        S.nav_button("Open these trades in the Trade Explorer", "trades", key="bt_to_trades", icon=":material/list_alt:")
    if c2.button("Save experiment files", key="bt_save", icon=":material/save:"):
        path = save_backtest(res, ds, S.instrument())
        st.success(f"Saved manifest, trades and metrics to {path}")
