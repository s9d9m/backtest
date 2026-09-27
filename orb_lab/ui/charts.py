"""Plotly figures for the dashboard, all in one visual style (see theme.py for the palette)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme as T

FONT = dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13, color=T.INK_2)


def style(fig: go.Figure, title: str = "", height: int = 340, legend: bool = False, yfmt: str | None = None, xtitle: str = "",
          ytitle: str = "", hover: str = "x unified") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=T.INK), x=0.0, xanchor="left", y=0.97) if title else None,
        height=height, margin=dict(l=10, r=14, t=46 if title else 16, b=10), paper_bgcolor="white", plot_bgcolor="white",
        font=FONT, showlegend=legend, hovermode=hover,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=T.BASELINE, ticks="outside", tickcolor=T.BASELINE, title_text=xtitle, automargin=True)
    fig.update_yaxes(gridcolor=T.GRID, zerolinecolor=T.BASELINE, linecolor=T.BASELINE, title_text=ytitle, automargin=True)
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    return fig


def equity(daily: pd.DataFrame, title: str = "Equity curve (net of costs)", start: float | None = None, gross: bool = False) -> go.Figure:
    fig = go.Figure()
    if gross and "gross_equity" in daily:
        fig.add_trace(go.Scatter(x=daily.index, y=daily["gross_equity"], name="Before costs", line=dict(color=T.NEUTRAL, width=1.5)))
    fig.add_trace(go.Scatter(x=daily.index, y=daily["net_equity"], name="Net equity", line=dict(color=T.BLUE, width=2.2),
                             hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra></extra>"))
    if start is not None:
        fig.add_hline(y=start, line_color=T.BASELINE, line_width=1)
    return style(fig, title, 360, legend=gross, yfmt="$,.0f")


def drawdown(daily: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    fig = go.Figure(go.Scatter(x=daily.index, y=daily["net_drawdown"], fill="tozeroy", line=dict(color=T.BAD, width=1.5),
                               fillcolor="rgba(208,59,59,0.15)", hovertemplate="%{x|%b %d, %Y}<br>%{y:.1%}<extra></extra>"))
    return style(fig, title, 240, yfmt=".0%")


def cum_r(trades: pd.DataFrame, title: str = "Cumulative R (trade by trade)") -> go.Figure:
    n = np.arange(1, len(trades) + 1)
    fig = go.Figure(go.Scatter(x=n, y=trades["r_multiple"].cumsum(), line=dict(color=T.BLUE, width=2.2),
                               hovertemplate="trade %{x}<br>%{y:+.2f}R<extra></extra>"))
    fig.add_hline(y=0, line_color=T.BASELINE, line_width=1)
    return style(fig, title, 300, xtitle="trade #", hover="closest")


def cum_r_time(trades: pd.DataFrame, title: str) -> go.Figure:
    x = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert("America/New_York")
    fig = go.Figure(go.Scatter(x=x, y=trades["r_multiple"].cumsum(), mode="lines+markers", line=dict(color=T.BLUE, width=2.2),
                               marker=dict(size=6), hovertemplate="%{x|%b %d %H:%M}<br>%{y:+.2f}R<extra></extra>"))
    fig.add_hline(y=0, line_color=T.BASELINE, line_width=1)
    return style(fig, title, 340, hover="closest", ytitle="cumulative R")


def trade_pnl(trades: pd.DataFrame, title: str = "Net P&L per trade") -> go.Figure:
    n = np.arange(1, len(trades) + 1)
    pnl = trades["net_pnl"].to_numpy()
    fig = go.Figure(go.Bar(x=n, y=pnl, marker_color=np.where(pnl >= 0, T.BLUE, T.BAD), marker_line_width=0,
                           hovertemplate="trade %{x}<br>$%{y:,.0f}<extra></extra>"))
    fig.update_layout(bargap=0.25)
    return style(fig, title, 300, yfmt="$,.0f", xtitle="trade #", hover="closest")


def position_size(trades: pd.DataFrame, unit: str, title: str | None = None) -> go.Figure:
    n = np.arange(1, len(trades) + 1)
    fig = go.Figure(go.Bar(x=n, y=trades["qty"], marker_color=T.BLUE, marker_line_width=0,
                           customdata=trades["risk_dollars"], hovertemplate=f"trade %{{x}}<br>%{{y:,.0f}} {unit}s<br>$%{{customdata:,.0f}} at risk<extra></extra>"))
    return style(fig, title or f"Position size ({unit}s)", 260, xtitle="trade #", hover="closest")


def hist(values, marker: float | None, title: str, xfmt: str | None = None, marker_label: str = "Historical",
         bins: int = 50, discrete: bool = False) -> go.Figure:
    values = np.asarray(values, dtype=float)
    if discrete:
        vc = pd.Series(values).value_counts().sort_index()
        fig = go.Figure(go.Bar(x=vc.index, y=vc.values / len(values), marker_color=T.NEUTRAL, marker_line_width=0,
                               hovertemplate="%{x}<br>%{y:.1%} of simulations<extra></extra>"))
        fig.update_yaxes(tickformat=".0%")
    else:
        fig = go.Figure(go.Histogram(x=values, nbinsx=bins, histnorm="probability", marker_color=T.NEUTRAL, marker_line_width=0,
                                     hovertemplate="%{x}<br>%{y:.1%} of simulations<extra></extra>"))
        fig.update_yaxes(tickformat=".0%")
    if marker is not None and np.isfinite(marker):
        fig.add_vline(x=marker, line_color=T.ORANGE, line_width=3)
        fig.add_annotation(x=marker, y=1.0, yref="paper", text=marker_label, showarrow=False, yanchor="bottom",
                           font=dict(color=T.INK, size=12), bgcolor="white")
    med = float(np.median(values)) if len(values) else None
    if med is not None:
        fig.add_vline(x=med, line_color=T.INK_2, line_width=1.5)
    fig.update_layout(bargap=0.05)
    style(fig, title, 330, hover="closest", ytitle="share of simulations")
    if xfmt:
        fig.update_xaxes(tickformat=xfmt)
    return fig


def sim_paths(paths: np.ndarray, historical: np.ndarray, title: str = "Simulated equity paths vs historical") -> go.Figure:
    fig = go.Figure()
    for p in paths:
        fig.add_trace(go.Scatter(y=p, mode="lines", line=dict(color="rgba(137,135,129,0.18)", width=1), showlegend=False, hoverinfo="skip"))
    q = np.percentile(paths, [5, 50, 95], axis=0) if len(paths) > 5 else None
    if q is not None:
        fig.add_trace(go.Scatter(y=q[1], mode="lines", line=dict(color=T.INK_2, width=1.5), name="Median simulation"))
    fig.add_trace(go.Scatter(y=historical, mode="lines", line=dict(color=T.ORANGE, width=3), name="Historical"))
    return style(fig, title, 380, legend=True, yfmt="$,.0f", xtitle="trade #", hover="closest")


def bars(labels, values, title: str, fmt: str = "+.2f", suffix: str = "R", highlight: int | None = None, muted: list[bool] | None = None,
         height: int = 300, horizontal: bool = False, text: list[str] | None = None, yfmt: str | None = None) -> go.Figure:
    values = np.asarray(values, dtype=float)
    colors = [T.BLUE if v >= 0 else T.BAD for v in values]
    if muted:
        colors = [T.NEUTRAL if m else c for c, m in zip(colors, muted)]
    if highlight is not None and 0 <= highlight < len(colors):
        colors[highlight] = T.ORANGE
    txt = text if text is not None else [f"{v:{fmt}}{suffix}" if np.isfinite(v) else "" for v in values]
    kw = dict(marker_color=colors, marker_line_width=0, text=txt, textposition="outside", cliponaxis=False,
              textfont=dict(size=11, color=T.INK_2))
    fig = go.Figure(go.Bar(y=list(map(str, labels)), x=values, orientation="h", **kw) if horizontal else go.Bar(x=list(map(str, labels)), y=values, **kw))
    if horizontal:
        fig.add_vline(x=0, line_color=T.BASELINE, line_width=1)
        fig.update_yaxes(autorange="reversed", type="category")
    else:
        fig.add_hline(y=0, line_color=T.BASELINE, line_width=1)
        fig.update_xaxes(type="category")
    fig.update_layout(bargap=0.35, uniformtext=dict(minsize=10, mode="hide"))
    style(fig, title, height, hover="closest")
    if yfmt:
        (fig.update_xaxes if horizontal else fig.update_yaxes)(tickformat=yfmt)
    return fig


def heatmap(table: pd.DataFrame, title: str, fmt: str = "+.2f", zmid: float = 0.0, height: int | None = None,
            mark: tuple[str, str] | None = None) -> go.Figure:
    z = table.to_numpy(dtype=float)
    finite = z[np.isfinite(z)]
    bound = float(np.max(np.abs(finite - zmid))) if finite.size else 1.0
    text = [[("" if not np.isfinite(v) else f"{v:{fmt}}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in table.columns], y=[str(i) for i in table.index], colorscale=T.DIVERGING,
                               zmid=zmid, zmin=zmid - bound, zmax=zmid + bound, text=text, texttemplate="%{text}",
                               textfont=dict(size=12), xgap=2, ygap=2, colorbar=dict(thickness=10, outlinewidth=0),
                               hovertemplate="%{y} × %{x}<br>%{z:.3f}<extra></extra>"))
    if mark is not None:
        fig.add_trace(go.Scatter(x=[str(mark[0])], y=[str(mark[1])], mode="markers", marker=dict(symbol="square-open", size=34,
                                 color=T.ORANGE, line=dict(width=3)), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(type="category", showgrid=False)
    fig.update_yaxes(type="category", showgrid=False, autorange="reversed")
    return style(fig, title, height or max(300, 36 * len(table) + 110), hover="closest")


def timeline(windows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {"train": "#d9d6cd", "val": "#f6cf6b", "oos": T.BLUE}
    names = {"train": "Train (not evidence)", "val": "Validation (not evidence)", "oos": "Blind OOS (evidence)"}
    for _, w in windows.iterrows():
        for phase in ("train", "val", "oos"):
            if f"{phase}_start" in w and pd.notna(w.get(f"{phase}_start")):
                a, b = pd.Timestamp(w[f"{phase}_start"]), pd.Timestamp(w[f"{phase}_end"]) + pd.Timedelta(days=1)
            else:
                x, y = str(w[f"{phase}_months"]).split("..")
                a, b = pd.Period(x, "M").start_time, pd.Period(y, "M").end_time
            fig.add_trace(go.Bar(x=[(b - a).total_seconds() * 1000], base=[a], y=[f"Fold {int(w['window']) + 1}"], orientation="h",
                                 marker_color=colors[phase], marker_line_width=0, name=names[phase], showlegend=bool(w["window"] == 0),
                                 hovertemplate=f"{names[phase]}: {a.date()} – {(b - pd.Timedelta(days=1)).date()}<extra></extra>"))
    fig.update_layout(barmode="overlay", bargap=0.35)
    fig.update_xaxes(type="date")
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Fold layout", max(220, 30 * len(windows) + 110), legend=True, hover="closest")


def trade_chart(prep, trade: pd.Series, title: str = "") -> go.Figure | None:
    """Candlesticks of the trade's session with the opening range, entry, stop, target and exit."""
    day = int(trade["day_idx"]) if "day_idx" in trade and pd.notna(trade.get("day_idx")) else None
    if day is None:
        dates = pd.DatetimeIndex(prep.dates.astype("datetime64[ns]"))
        pos = np.where(dates == pd.Timestamp(trade["session_date"]).normalize().tz_localize(None))[0]
        if not len(pos):
            return None
        day = int(pos[0])
    if day >= prep.n_days or str(prep.dates[day]) != str(pd.Timestamp(trade["session_date"]).date()):
        return None
    lo, hi = int(prep.day_start[day]), int(prep.day_end[day])
    tick = prep.tick_size
    ts = prep.timestamps(np.full(hi - lo, day), prep.tod[lo:hi]).tz_convert("America/New_York")
    fig = go.Figure(go.Candlestick(x=ts, open=prep.open[lo:hi] * tick, high=prep.high[lo:hi] * tick, low=prep.low[lo:hi] * tick,
                                   close=prep.close[lo:hi] * tick, increasing_line_color="#8a99ad", decreasing_line_color="#3b4452",
                                   increasing_fillcolor="#c9d3e0", decreasing_fillcolor="#3b4452", name="price", line=dict(width=1)))
    if pd.notna(trade.get("or_high")):
        fig.add_hrect(y0=trade["or_low"], y1=trade["or_high"], fillcolor="rgba(250,178,25,0.16)", line_width=0)
    for y, color, label in ((trade.get("stop_price"), T.BAD, "stop"), (trade.get("target_price"), GOOD_LINE, "target")):
        if pd.notna(y):
            fig.add_hline(y=y, line_color=color, line_width=1.5, annotation_text=label, annotation_position="top left",
                          annotation_font=dict(color=color, size=12))
    et = pd.Timestamp(trade["entry_time"]).tz_convert("America/New_York")
    xt = pd.Timestamp(trade["exit_time"]).tz_convert("America/New_York")
    up = trade["direction"] == "long"
    fig.add_trace(go.Scatter(x=[et], y=[trade["entry_price"]], mode="markers", name="entry",
                             marker=dict(symbol="triangle-up" if up else "triangle-down", size=14, color=T.BLUE, line=dict(color="white", width=2))))
    fig.add_trace(go.Scatter(x=[xt], y=[trade["exit_price"]], mode="markers", name="exit",
                             marker=dict(symbol="x", size=12, color=T.ORANGE, line=dict(color="white", width=1))))
    first = max(0, int(np.searchsorted(ts, et - pd.Timedelta(minutes=45))))
    last = min(len(ts) - 1, int(np.searchsorted(ts, xt + pd.Timedelta(minutes=45))))
    fig.update_xaxes(range=[ts[first], ts[last]], rangeslider_visible=False)
    return style(fig, title, 460, legend=True, hover="closest")


GOOD_LINE = "#0a8a0a"
