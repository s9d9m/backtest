"""Baselines and controls for PHASE0_FREE_PROXY.

* **Random-direction control.** For every ORB trade, keep the same entry bar, entry price, risk
  distance and target multiple, but choose the direction by coin flip. Both directions are simulated
  once with the engine's exit rules (conservative ambiguity, one-tick trade-through targets, slippage
  on stop/time exits, time exit at the session exit), then many random assignments are drawn. This
  isolates whether the breakout *direction* carries information beyond the timing and risk structure.
* **Buy-and-hold.** First open to last close of the split.
* **Intraday long.** Buy the 09:30 open and sell the 15:55 open every session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.execution import EPS
from ..engine.session_data import PreparedData


def _bar_index(prep: PreparedData, day: int, tod: int) -> int:
    lo, hi = int(prep.day_start[day]), int(prep.day_end[day])
    idx = np.searchsorted(prep.tod[lo:hi], tod)
    return lo + int(idx)


def simulate_fixed_trade(prep: PreparedData, day: int, entry_idx: int, entry_theo: float, direction: int, risk: float,
                         target_r: float, slip: float, intrabar: bool, exit_min: int) -> tuple[float, str]:
    """Outcome (R net of slippage, exit reason) of a trade with a fixed entry and stop/target distances.

    Prices in tick units. Mirrors the kernel's conservative rules (see orb_lab.engine.execution).
    """
    s = direction
    e = entry_theo + s * slip
    stop_n = np.floor(s * e - risk + EPS)  # long-equivalent coordinates, rounded away from entry
    risk_n = s * e - stop_n
    tgt_n = np.ceil(s * e + target_r * risk_n - EPS) if target_r > 0 else np.inf
    end = int(prep.day_end[day])
    E = s * e
    for i in range(entry_idx, end):
        if prep.tod[i] >= exit_min:
            x = s * prep.open[i] - slip
            return (x - E) / risk_n, "time"
        if s > 0:
            fo, fh, fl = prep.open[i], prep.high[i], prep.low[i]
        else:
            fo, fh, fl = -prep.open[i], -prep.low[i], -prep.high[i]
        entry_bar_intra = i == entry_idx and intrabar
        if not entry_bar_intra:
            if fo <= stop_n + EPS:
                return (fo - slip - E) / risk_n, "stop"
            if fo >= tgt_n + 1.0 - EPS:
                return (tgt_n - E) / risk_n, "target"
        s_hit = fl <= stop_n + EPS
        t_hit = fh >= tgt_n + 1.0 - EPS and not entry_bar_intra
        if s_hit:
            return (stop_n - slip - E) / risk_n, "stop"
        if t_hit:
            return (tgt_n - E) / risk_n, "target"
    x = s * prep.close[end - 1] - slip
    return (x - E) / risk_n, "data_end"


def both_direction_outcomes(prep: PreparedData, trades: pd.DataFrame, target_r: float, slip: float, exit_min_by_day: np.ndarray,
                            cost_ticks_round_trip: float = 0.0) -> pd.DataFrame:
    """R outcome of each trade taken long and short (net of ``slip`` and a fixed round-trip cost in ticks)."""
    tick = prep.tick_size
    rows = []
    for _, t in trades.iterrows():
        day = int(t["day_idx"])
        tod = t["entry_time"].hour * 60 + t["entry_time"].minute
        idx = _bar_index(prep, day, tod)
        entry_theo = t["entry_price"] / tick - t["dir"] * t["entry_slip_ticks"]
        risk = float(t["risk_ticks"])
        out = {}
        for d, name in ((1, "long"), (-1, "short")):
            out[name] = simulate_fixed_trade(prep, day, idx, entry_theo, d, risk, target_r, slip, bool(t["intrabar_entry"]),
                                             int(exit_min_by_day[day]))[0] - cost_ticks_round_trip / risk
        rows.append({"trade_id": t["trade_id"], "actual_dir": int(t["dir"]), "r_long": out["long"], "r_short": out["short"],
                     "actual_r_sim": out["long"] if t["dir"] > 0 else out["short"]})
    return pd.DataFrame(rows)


def random_direction_control(outcomes: pd.DataFrame, n_sims: int = 10_000, seed: int = 7) -> dict:
    if outcomes.empty:
        return {"n_trades": 0}
    rng = np.random.default_rng(seed)
    flips = rng.integers(0, 2, size=(n_sims, len(outcomes)))
    r = np.where(flips == 1, outcomes["r_long"].to_numpy()[None, :], outcomes["r_short"].to_numpy()[None, :])
    means = r.mean(axis=1)
    actual = float(outcomes["actual_r_sim"].mean())
    return {
        "n_trades": int(len(outcomes)),
        "n_sims": n_sims,
        "seed": seed,
        "strategy_mean_r": actual,
        "random_mean_r_median": float(np.median(means)),
        "random_mean_r_p05": float(np.quantile(means, 0.05)),
        "random_mean_r_p95": float(np.quantile(means, 0.95)),
        "p_value_random_ge_strategy": float(np.mean(means >= actual - 1e-12)),
    }


def buy_and_hold(prep: PreparedData, d_lo: int, d_hi: int, shares: float) -> dict:
    if d_hi <= d_lo:
        return {}
    tick = prep.tick_size
    first = prep.open[int(prep.day_start[d_lo])] * tick
    last = prep.close[int(prep.day_end[d_hi - 1]) - 1] * tick
    intraday = []
    for d in range(d_lo, d_hi):
        lo, hi = int(prep.day_start[d]), int(prep.day_end[d])
        tod = prep.tod[lo:hi]
        exit_idx = np.searchsorted(tod, prep.session_exit_min[d])
        px_exit = prep.open[lo + exit_idx] if lo + exit_idx < hi else prep.close[hi - 1]
        intraday.append((px_exit - prep.open[lo]) * tick)
    intraday = np.array(intraday)
    return {
        "buy_hold_return_pct": float(last / first - 1) * 100,
        "buy_hold_pnl_per_shares": float((last - first) * shares),
        "intraday_long_mean_per_share": float(intraday.mean()),
        "intraday_long_total_per_shares": float(intraday.sum() * shares),
        "intraday_long_win_rate": float((intraday > 0).mean()),
        "sessions": int(d_hi - d_lo),
    }
