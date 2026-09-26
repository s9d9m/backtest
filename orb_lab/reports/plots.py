"""Plotly figure builders shared by the dashboard and reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NET = "#2a6fdb"
GROSS = "#9aa5b1"
LONG = "#1b9e77"
SHORT = "#d95f02"
NEG = "#c0392b"
LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40), hovermode="x unified")


def equity_and_drawdown(daily: pd.DataFrame, benchmark: pd.DataFrame | None = None, title: str = "Equity (net) and drawdown") -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["net_equity"], name="Net equity", line=dict(color=NET)), 1, 1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["gross_equity"], name="Gross (before costs)", line=dict(color=GROSS, dash="dot")), 1, 1)
    if benchmark is not None and len(benchmark):
        fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark["net_equity"], name="Frictionless benchmark", line=dict(color="#7f8c8d", dash="dash")), 1, 1)
    fig.add_trace(
        go.Scatter(x=daily.index, y=daily["net_drawdown"] * 100, name="Drawdown %", fill="tozeroy", line=dict(color=NEG)), 2, 1
    )
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="DD %", row=2, col=1)
    fig.update_layout(title=title, height=520, **LAYOUT)
    return fig


def monthly_returns_heatmap(daily: pd.DataFrame) -> go.Figure:
    rets = (1 + daily["net_return"]).groupby([daily.index.year, daily.index.month]).prod() - 1
    table = rets.unstack(level=1).reindex(columns=range(1, 13)) * 100
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    bound = float(np.nanmax(np.abs(table.to_numpy()))) if table.size and np.isfinite(table.to_numpy()).any() else 1.0
    fig = go.Figure(
        go.Heatmap(
            z=table.to_numpy(),
            x=months,
            y=[str(y) for y in table.index],
            colorscale="RdBu",
            zmid=0,
            zmin=-bound,
            zmax=bound,
            text=np.where(np.isnan(table.to_numpy()), "", np.round(table.to_numpy(), 1).astype(str)),
            texttemplate="%{text}",
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(title="Monthly net returns (%)", height=max(250, 40 * len(table) + 120), **{k: v for k, v in LAYOUT.items() if k != "hovermode"})
    return fig


def annual_returns(daily: pd.DataFrame) -> go.Figure:
    rets = ((1 + daily["net_return"]).groupby(daily.index.year).prod() - 1) * 100
    fig = go.Figure(go.Bar(x=[str(y) for y in rets.index], y=rets.values, marker_color=[NET if v >= 0 else NEG for v in rets.values]))
    fig.update_layout(title="Annual net returns (%)", height=320, **LAYOUT)
    return fig


def r_distribution(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for side, color in (("long", LONG), ("short", SHORT)):
        sub = trades[trades["direction"] == side]
        if len(sub):
            fig.add_trace(go.Histogram(x=sub["r_multiple"], name=side, marker_color=color, opacity=0.65, xbins=dict(size=0.1)))
    fig.update_layout(barmode="overlay", title="R-multiple distribution (net)", xaxis_title="R", height=340, **LAYOUT)
    return fig


def pnl_histogram(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(x=trades["net_pnl"], marker_color=NET, nbinsx=60))
    fig.update_layout(title="Trade net P&L", xaxis_title="$", height=340, **LAYOUT)
    return fig


def rolling_metrics(daily: pd.DataFrame, trades: pd.DataFrame, window_days: int = 126, window_trades: int = 50) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.08, subplot_titles=(
        f"Rolling Sharpe ({window_days} sessions)", f"Rolling expectancy ({window_trades} trades, R)", f"Rolling max drawdown ({window_days} sessions, %)"))
    r = daily["net_return"]
    sharpe = r.rolling(window_days).mean() / r.rolling(window_days).std() * np.sqrt(252)
    fig.add_trace(go.Scatter(x=daily.index, y=sharpe, name="Sharpe", line=dict(color=NET)), 1, 1)
    if len(trades):
        exp = trades["r_multiple"].rolling(window_trades).mean()
        fig.add_trace(go.Scatter(x=trades["exit_time"], y=exp, name="Expectancy R", line=dict(color=LONG)), 2, 1)
    eq = daily["net_equity"]
    roll_dd = (eq / eq.rolling(window_days, min_periods=1).max() - 1).rolling(window_days, min_periods=1).min() * 100
    fig.add_trace(go.Scatter(x=daily.index, y=roll_dd, name="Rolling DD", line=dict(color=NEG)), 3, 1)
    fig.update_layout(height=700, showlegend=False, **LAYOUT)
    return fig


def long_short_equity(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for side, color in (("long", LONG), ("short", SHORT)):
        sub = trades[(trades["direction"] == side) & (~trades["sized_out"])]
        if len(sub):
            fig.add_trace(go.Scatter(x=sub["exit_time"], y=sub["net_pnl"].cumsum(), name=side, line=dict(color=color)))
    fig.update_layout(title="Cumulative net P&L by direction", height=340, **LAYOUT)
    return fig


def breakdown_bar(trades: pd.DataFrame, by: str, title: str) -> go.Figure:
    grouped = trades.groupby(by, observed=True)["net_pnl"].agg(["sum", "count", "mean"])
    fig = go.Figure(go.Bar(x=[str(i) for i in grouped.index], y=grouped["sum"], marker_color=[NET if v >= 0 else NEG for v in grouped["sum"]],
                           customdata=np.stack([grouped["count"], grouped["mean"]], axis=1),
                           hovertemplate="%{x}<br>net %{y:,.0f}<br>trades %{customdata[0]}<br>avg %{customdata[1]:,.1f}<extra></extra>"))
    fig.update_layout(title=title, height=320, **{k: v for k, v in LAYOUT.items() if k != "hovermode"})
    return fig


def heatmap(table: pd.DataFrame, title: str, metric: str, zmid: float | None = 0.0) -> go.Figure:
    z = table.to_numpy(dtype=float)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(c) for c in table.columns],
            y=[str(i) for i in table.index],
            colorscale="RdBu",
            zmid=zmid,
            text=np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(z))),
            texttemplate="%{text}",
            colorbar=dict(title=metric),
        )
    )
    fig.update_layout(title=title, xaxis_title=table.columns.name, yaxis_title=table.index.name, height=max(320, 32 * len(table) + 140),
                      **{k: v for k, v in LAYOUT.items() if k != "hovermode"})
    return fig
