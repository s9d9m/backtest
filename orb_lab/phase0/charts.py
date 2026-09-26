"""Visual verification of individual trades on the actual market bars (PNG via matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..engine.params import StrategyParams  # noqa: E402
from ..engine.session_data import PreparedData  # noqa: E402

UP, DOWN = "#1b9e77", "#c0392b"


def chart_trade(prep: PreparedData, trade: pd.Series, params: StrategyParams, symbol: str, label: str, path: Path) -> Path:
    day = int(trade["day_idx"])
    lo, hi = int(prep.day_start[day]), int(prep.day_end[day])
    tick = prep.tick_size
    ts = prep.timestamps(np.full(hi - lo, day), prep.tod[lo:hi]).tz_localize(None)
    o, h, l, c = (arr[lo:hi] * tick for arr in (prep.open, prep.high, prep.low, prep.close))
    exit_t = trade["exit_time"].tz_localize(None)
    view_end = exit_t + pd.Timedelta(minutes=45)
    keep = ts <= view_end
    ts, o, h, l, c = ts[keep], o[keep], h[keep], l[keep], c[keep]
    bar_w = pd.Timedelta(minutes=prep.base_minutes) * 0.7

    fig, ax = plt.subplots(figsize=(13, 6.5))
    start = pd.Timestamp(trade["session_date"]).normalize() + pd.Timedelta(minutes=params.orb_start_min)
    rng_end = start + pd.Timedelta(minutes=params.range_minutes)
    ax.axvspan(start, rng_end, color="#f2b134", alpha=0.18, label=f"opening range {params.range_minutes}m")
    ax.hlines([trade["or_high"], trade["or_low"]], start, ts[-1], colors="#b07d12", linestyles="-", linewidth=1.1)
    ax.hlines(0.5 * (trade["or_high"] + trade["or_low"]), start, ts[-1], colors="#b07d12", linestyles=":", linewidth=0.8)
    for t_, o_, h_, l_, c_ in zip(ts, o, h, l, c):
        col = UP if c_ >= o_ else DOWN
        ax.vlines(t_ + bar_w / 2, l_, h_, color=col, linewidth=0.9)
        ax.add_patch(plt.Rectangle((mdates.date2num(t_), min(o_, c_)), mdates.date2num(t_ + bar_w) - mdates.date2num(t_),
                                   max(abs(c_ - o_), tick * 0.3), color=col, alpha=0.85))
    entry_t = trade["entry_time"].tz_localize(None)
    sig_t = trade["signal_time"].tz_localize(None)
    if params.entry_method != "stop":
        ax.axvspan(sig_t - pd.Timedelta(minutes=params.entry_tf), sig_t, color="#2a6fdb", alpha=0.10, label=f"signal bar ({params.entry_tf}m close)")
    ax.hlines(trade["stop_price"], entry_t, exit_t + bar_w, colors=DOWN, linestyles="--", linewidth=1.3, label=f"stop {trade['stop_price']:.2f}")
    if pd.notna(trade["target_price"]):
        ax.hlines(trade["target_price"], entry_t, exit_t + bar_w, colors=UP, linestyles="--", linewidth=1.3, label=f"target {trade['target_price']:.2f}")
    marker = "^" if trade["dir"] > 0 else "v"
    ax.scatter([entry_t + bar_w / 2], [trade["entry_price"]], marker=marker, s=160, color="#2a6fdb", zorder=5, edgecolor="black",
               label=f"entry {trade['direction']} {trade['entry_price']:.2f} @ {entry_t:%H:%M}")
    ax.scatter([exit_t + bar_w / 2], [trade["exit_price"]], marker="X", s=160, color="black", zorder=5,
               label=f"exit {trade['exit_reason']} {trade['exit_price']:.2f} @ {exit_t:%H:%M}")
    ax.axvline(start, color="#555", linewidth=0.8)
    ax.text(start, ax.get_ylim()[1], " 09:30", va="top", fontsize=8, color="#555")
    ax.set_title(f"{symbol} {pd.Timestamp(trade['session_date']).date()} | {label} | R = {trade['r_multiple']:+.2f} | "
                 f"{params.range_minutes}m OR, {params.entry_method}, {params.stop_method}{'' if not params.stop_param else ':' + format(params.stop_param, 'g')}, "
                 f"{params.target_r}R, cutoff {params.cutoff}\nFREE PROXY EXPERIMENT — NOT FUTURES VALIDATION", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("price ($)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def heatmap_png(table: pd.DataFrame, title: str, path: Path, fmt: str = "{:+.2f}") -> Path:
    fig, ax = plt.subplots(figsize=(max(6, 0.75 * table.shape[1] + 2), max(3, 0.5 * table.shape[0] + 1.5)))
    z = table.to_numpy(dtype=float)
    bound = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
    im = ax.imshow(z, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index], fontsize=8)
    ax.set_xlabel(str(table.columns.name))
    ax.set_ylabel(str(table.index.name))
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isfinite(z[i, j]):
                ax.text(j, i, fmt.format(z[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
