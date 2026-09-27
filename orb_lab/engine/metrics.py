"""Performance metrics.

Daily statistics (Sharpe, Sortino, volatility, drawdown) are computed over **every eligible session**
in the tested range, including days without a trade. Annualisation uses 252 sessions/year. CAGR uses
the calendar span between the first and last session.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def _streaks(signs: np.ndarray) -> tuple[int, int]:
    best_win = best_loss = cur_win = cur_loss = 0
    for s in signs:
        if s > 0:
            cur_win += 1
            cur_loss = 0
        elif s < 0:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = cur_loss = 0
        best_win = max(best_win, cur_win)
        best_loss = max(best_loss, cur_loss)
    return best_win, best_loss


def drawdown_stats(equity: np.ndarray, starting_equity: float) -> tuple[float, int]:
    """Max drawdown (fraction, negative) and longest drawdown duration (sessions below prior peak)."""
    if len(equity) == 0:
        return 0.0, 0
    peak = np.maximum.accumulate(np.concatenate(([starting_equity], equity)))[1:]
    dd = equity / peak - 1.0
    under = dd < -1e-12
    longest = cur = 0
    for flag in under:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return float(dd.min()), int(longest)


def _years(dates: pd.DatetimeIndex) -> float:
    if len(dates) < 2:
        return max(len(dates), 1) / PERIODS_PER_YEAR
    return max((dates[-1] - dates[0]).days / 365.25, 1.0 / PERIODS_PER_YEAR)


def return_stats(daily_pnl: np.ndarray, dates: pd.DatetimeIndex, starting_equity: float) -> dict:
    equity = starting_equity + np.cumsum(daily_pnl)
    prev = np.concatenate(([starting_equity], equity[:-1]))
    rets = np.where(prev > 0, daily_pnl / np.where(prev > 0, prev, 1.0), 0.0)
    years = _years(dates)
    end = equity[-1] if len(equity) else starting_equity
    total_return = end / starting_equity - 1.0
    cagr = (end / starting_equity) ** (1.0 / years) - 1.0 if end > 0 else -1.0
    std = rets.std(ddof=1) if len(rets) > 1 else 0.0
    mean = rets.mean() if len(rets) else 0.0
    downside = math.sqrt(np.mean(np.minimum(rets, 0.0) ** 2)) if len(rets) else 0.0
    sharpe = mean / std * math.sqrt(PERIODS_PER_YEAR) if std > 0 else 0.0
    sortino = mean / downside * math.sqrt(PERIODS_PER_YEAR) if downside > 0 else 0.0
    max_dd, dd_dur = drawdown_stats(equity, starting_equity)
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "start_equity": starting_equity,
        "end_equity": float(end),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_vol": float(std * math.sqrt(PERIODS_PER_YEAR)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "max_dd_duration_days": dd_dur,
        "calmar": float(calmar),
        "years": float(years),
    }


def trade_stats(pnl: np.ndarray, r: np.ndarray, hold_min: np.ndarray, years: float) -> dict:
    n = len(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    win_streak, loss_streak = _streaks(np.sign(pnl))
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    return {
        "n_trades": int(n),
        "trades_per_year": float(n / years) if years > 0 else 0.0,
        "net_pnl": float(pnl.sum()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "expectancy": float(pnl.mean()) if n else 0.0,
        "win_rate": float(len(wins) / n) if n else 0.0,
        "loss_rate": float(len(losses) / n) if n else 0.0,
        "avg_winner": avg_win,
        "avg_loser": avg_loss,
        "payoff_ratio": float(avg_win / abs(avg_loss)) if avg_loss < 0 else 0.0,
        "avg_r": float(np.mean(r)) if n else 0.0,
        "median_r": float(np.median(r)) if n else 0.0,
        "std_r": float(np.std(r, ddof=1)) if n > 1 else 0.0,
        "t_stat_r": float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(n)) if n > 1 and np.std(r, ddof=1) > 0 else 0.0,
        "longest_win_streak": int(win_streak),
        "longest_loss_streak": int(loss_streak),
        "best_trade": float(pnl.max()) if n else 0.0,
        "worst_trade": float(pnl.min()) if n else 0.0,
        "avg_hold_min": float(np.mean(hold_min)) if n else 0.0,
        "median_hold_min": float(np.median(hold_min)) if n else 0.0,
    }


def concentration_stats(pnl: np.ndarray, years_of_trade: np.ndarray) -> dict:
    """How concentrated are profits? (top-5 trade share, single-year share, % profitable years)."""
    total = pnl.sum()
    out = {"top5_share": float("nan"), "max_year_share": float("nan"), "pct_years_profitable": 0.0, "pnl_ex_top5": float("nan")}
    if len(pnl) == 0:
        return out
    top5 = np.sort(pnl)[::-1][:5]
    out["pnl_ex_top5"] = float(total - top5[top5 > 0].sum())
    if total > 0:
        out["top5_share"] = float(top5[top5 > 0].sum() / total)
    by_year = pd.Series(pnl).groupby(years_of_trade).sum()
    if total > 0 and (by_year > 0).any():
        out["max_year_share"] = float(by_year.max() / total)
    out["pct_years_profitable"] = float((by_year > 0).mean()) if len(by_year) else 0.0
    return out


def compute_metrics(trades: pd.DataFrame, daily: pd.DataFrame, starting_equity: float, kind: str = "net") -> dict:
    """Full metric set for a sized trade log and its daily equity frame (``kind`` = net | gross)."""
    executed = trades[~trades["sized_out"]] if len(trades) and "sized_out" in trades else trades
    dates = pd.DatetimeIndex(daily.index)
    daily_pnl = daily[f"{kind}_pnl"].to_numpy() if len(daily) else np.zeros(0)
    out = return_stats(daily_pnl, dates, starting_equity)
    pnl_col = "net_pnl" if kind == "net" else "gross_pnl"
    pnl = executed[pnl_col].to_numpy() if len(executed) else np.zeros(0)
    if len(executed):
        risk = executed["risk_dollars"].to_numpy()
        r = np.where(risk > 0, pnl / np.where(risk > 0, risk, 1.0), 0.0)
        hold = executed["holding_minutes"].to_numpy()
        years_of_trade = pd.DatetimeIndex(executed["session_date"]).year.to_numpy()
    else:
        r = hold = years_of_trade = np.zeros(0)
    out.update(trade_stats(pnl, r, hold, out["years"]))
    out.update(concentration_stats(pnl, years_of_trade))
    def total(col: str) -> float:
        return float(executed[col].sum()) if len(executed) and col in executed else 0.0

    out["commission_paid"] = total("commission")
    out["fees_paid"] = total("fees")
    out["friction_paid"] = total("friction")
    out["slippage_cost"] = total("slippage_cost")
    out["total_cost"] = total("total_cost") if len(executed) and "total_cost" in executed else out["commission_paid"] + out["slippage_cost"]
    out["cost_per_trade"] = out["total_cost"] / len(executed) if len(executed) else 0.0
    out["n_long"] = int((executed["direction"] == "long").sum()) if len(executed) else 0
    out["n_short"] = int((executed["direction"] == "short").sum()) if len(executed) else 0
    for side in ("long", "short"):
        sub = executed[executed["direction"] == side] if len(executed) else executed
        out[f"pnl_{side}"] = float(sub[pnl_col].sum()) if len(sub) else 0.0
        out[f"expectancy_r_{side}"] = float(sub["r_multiple"].mean()) if len(sub) and "r_multiple" in sub else 0.0
    out["ambiguous_exit_pct"] = float(executed["ambiguous_exit"].mean()) if len(executed) and "ambiguous_exit" in executed else 0.0
    out["cum_r"] = float(r.sum()) if len(r) else 0.0
    out["gross_pnl"] = float(executed["gross_pnl"].sum()) if len(executed) else 0.0
    out["n_sized_out"] = int(trades["sized_out"].sum()) if len(trades) and "sized_out" in trades else 0
    return out


def fast_summary(
    net_pc: np.ndarray,
    r: np.ndarray,
    day_pos: np.ndarray,
    direction: np.ndarray,
    trade_year: np.ndarray,
    hold_min: np.ndarray,
    dates: pd.DatetimeIndex,
    starting_equity: float,
) -> dict:
    """Metrics for a fixed 1-contract run, without building DataFrames (used by grid search).

    Matches :func:`compute_metrics` for ``SizingParams(mode="fixed_contracts", contracts=1)``.
    """
    n_days = len(dates)
    daily = np.bincount(day_pos, weights=net_pc, minlength=n_days) if len(net_pc) else np.zeros(n_days)
    out = return_stats(daily, dates, starting_equity)
    out.update(trade_stats(net_pc, r, hold_min, out["years"]))
    out.update(concentration_stats(net_pc, trade_year))
    for sign, name in ((1, "long"), (-1, "short")):
        mask = direction == sign
        sub = net_pc[mask]
        out[f"n_{name}"] = int(mask.sum())
        out[f"pnl_{name}"] = float(sub.sum())
        out[f"expectancy_r_{name}"] = float(r[mask].mean()) if mask.any() else 0.0
        out[f"win_rate_{name}"] = float((sub > 0).mean()) if mask.any() else 0.0
    return out
