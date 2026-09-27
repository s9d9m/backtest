"""Pure (non-Streamlit) helpers behind the dashboard's summaries, kept here so they can be unit tested."""

from __future__ import annotations

import numpy as np
import pandas as pd

STABILITY_LABEL = {"plateau": "Broad plateau", "mixed": "Moderate", "spike": "Spike / fragile", "isolated": "Isolated",
                   "not_positive": "Not positive"}


def best_historical(results: pd.DataFrame) -> pd.Series | None:
    """The single highest in-sample net P&L (what a naive optimiser would pick)."""
    if results is None or results.empty:
        return None
    return results.sort_values(["net_pnl", "config_key"], ascending=[False, True]).iloc[0]


def robust_candidate(results: pd.DataFrame, min_trades: int | None = None) -> tuple[pd.Series | None, str]:
    """Best composite-ranked configuration whose neighbours also work (plateau) and that has enough trades.

    Falls back to 'mixed' stability, and returns (None, reason) if nothing qualifies. Requires the columns added by
    ``grid_stability`` (``stability``, ``nbr_median``) and ``rank_composite``.
    """
    if results is None or results.empty or "stability" not in results:
        return None, "no optimisation results"
    if min_trades is None:
        min_trades = int(max(20, np.nanpercentile(results["n_trades"], 25)))
    ok = results[(results["n_trades"] >= min_trades) & (results["avg_r"] > 0)]
    for level in ("plateau", "mixed"):
        pool = ok[ok["stability"] == level]
        if len(pool):
            row = pool.sort_values(["rank_composite", "config_key"]).iloc[0]
            why = ("neighbouring settings keep most of the result" if level == "plateau" else
                   "no broad plateau exists; this is the best 'moderate' configuration")
            return row, f"{why}; at least {min_trades} trades"
    return None, f"no configuration with positive expectancy, ≥ {min_trades} trades and stable neighbours"


def fold_consistency(windows: pd.DataFrame) -> dict:
    traded = windows[windows.get("selected", pd.Series(dtype=int)) >= 0] if "selected" in windows else windows
    if not len(traded) or "oos_exp_r" not in traded:
        return {"folds": 0, "positive": 0, "share": float("nan")}
    pos = int((traded["oos_exp_r"] > 0).sum())
    return {"folds": int(len(traded)), "positive": pos, "share": pos / len(traded)}


def degradation(summary: dict) -> dict:
    """Median expectancy by phase and the relative drop from training to OOS."""
    tr, va, oos = (summary.get(k) for k in ("median_train_exp_r", "median_val_exp_r", "median_oos_exp_r"))
    drop = (1 - oos / tr) if (tr is not None and oos is not None and tr > 0) else float("nan")
    return {"train": tr, "validation": va, "oos": oos, "drop_share": drop}


def mc_headline(result) -> dict:
    """The numbers shown at the top of the Monte Carlo page."""
    sims, hist, cfg = result.sims, result.historical, result.config
    return {
        "hist_end": hist["ending_equity"], "hist_return": hist["total_return"], "median_end": float(sims["ending_equity"].median()),
        "p_loss": result.probabilities["p_loss"], "hist_dd": hist["max_drawdown"], "median_dd": float(sims["max_drawdown"].median()),
        "p5_end": float(sims["ending_equity"].quantile(0.05)), "p5_dd": float(sims["max_drawdown"].quantile(0.05)),
        "p95_end": float(sims["ending_equity"].quantile(0.95)), "median_streak": float(sims["longest_losing_streak"].median()),
        "p95_streak": float(sims["longest_losing_streak"].quantile(0.95)), "start": cfg.starting_equity,
    }


def filter_trades(trades: pd.DataFrame, *, start=None, end=None, sides=None, outcome: str = "all", methods=None, markets=None,
                  r_range: tuple[float, float] | None = None) -> pd.DataFrame:
    t = trades
    if t is None or t.empty:
        return t
    if "sized_out" in t:
        t = t[~t["sized_out"].astype(bool)]
    d = pd.to_datetime(t["session_date"])
    if start is not None:
        t, d = t[d >= pd.Timestamp(start)], d[d >= pd.Timestamp(start)]
    if end is not None:
        t = t[d <= pd.Timestamp(end)]
    if sides:
        t = t[t["direction"].isin(sides)]
    if outcome == "winners":
        t = t[t["net_pnl"] > 0]
    elif outcome == "losers":
        t = t[t["net_pnl"] <= 0]
    if methods and "entry_method" in t:
        t = t[t["entry_method"].isin(methods)]
    if markets and "market" in t:
        t = t[t["market"].isin(markets)]
    if r_range is not None:
        t = t[(t["r_multiple"] >= r_range[0] - 1e-12) & (t["r_multiple"] <= r_range[1] + 1e-12)]
    return t
