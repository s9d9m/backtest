"""Ranking objectives for optimization results.

The composite score is an *in-sample quality* measure. It rewards risk-adjusted return and
statistical strength, and penalises low trade counts, profit concentration (a few trades or a single
year) and deep drawdowns. It deliberately does not reward raw return. Neighbourhood/plateau terms
and out-of-sample terms are added by the robustness and walk-forward milestones.

Several alternative objectives are always reported so you can see whether conclusions depend on the
choice of objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"sharpe": 0.30, "t_stat_r": 0.25, "log_profit_factor": 0.15, "calmar": 0.10, "pct_years_profitable": 0.20}
DEFAULT_PENALTIES = {"min_trades": 100, "max_top5_share": 0.50, "max_year_share": 0.50, "max_drawdown": 0.30}

ALTERNATIVE_OBJECTIVES = {
    "net_pnl": "Highest net P&L (return-chasing; for reference only)",
    "sharpe": "Daily Sharpe ratio",
    "expectancy_r": "Mean R per trade",
    "t_stat_r": "t-statistic of mean R (penalises small samples)",
    "profit_factor": "Profit factor",
    "calmar": "CAGR / max drawdown",
    "composite": "Composite in-sample quality score",
}


@dataclass
class ObjectiveConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    penalties: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PENALTIES))

    @classmethod
    def from_dict(cls, data: dict | None) -> "ObjectiveConfig":
        data = data or {}
        return cls(weights={**DEFAULT_WEIGHTS, **(data.get("weights") or {})}, penalties={**DEFAULT_PENALTIES, **(data.get("penalties") or {})})

    def to_dict(self) -> dict:
        return {"weights": dict(self.weights), "penalties": dict(self.penalties)}


def composite_score(frame: pd.DataFrame, config: ObjectiveConfig | None = None) -> pd.Series:
    """Composite score per row. Components are clipped to comparable ranges before weighting."""
    cfg = config or ObjectiveConfig()
    w = cfg.weights
    p = cfg.penalties
    pf = frame["profit_factor"].replace([np.inf], 10.0).clip(lower=1e-3)
    comps = {
        "sharpe": frame["sharpe"].clip(-3, 3) / 3,
        "t_stat_r": frame["t_stat_r"].clip(-5, 5) / 5,
        "log_profit_factor": np.log(pf).clip(-1, 1),
        "calmar": frame["calmar"].clip(-3, 3) / 3,
        "pct_years_profitable": frame["pct_years_profitable"] * 2 - 1,
    }
    score = sum(w.get(k, 0.0) * v for k, v in comps.items())
    total_w = sum(abs(w.get(k, 0.0)) for k in comps) or 1.0
    score = score / total_w
    # multiplicative penalties only shrink positive scores towards zero (and deepen negative ones)
    factor = pd.Series(1.0, index=frame.index)
    min_trades = p.get("min_trades", 0)
    if min_trades:
        factor *= (frame["n_trades"] / min_trades).clip(upper=1.0)
    if p.get("max_top5_share"):
        excess = (frame["top5_share"].fillna(1.0) - p["max_top5_share"]).clip(lower=0)
        factor *= (1 - excess).clip(lower=0)
    if p.get("max_year_share"):
        excess = (frame["max_year_share"].fillna(1.0) - p["max_year_share"]).clip(lower=0)
        factor *= (1 - excess).clip(lower=0)
    if p.get("max_drawdown"):
        excess = (frame["max_dd"].abs() - p["max_drawdown"]).clip(lower=0)
        factor *= (1 - 2 * excess).clip(lower=0)
    return pd.Series(np.where(score > 0, score * factor, score * (2 - factor)), index=frame.index)


def add_objectives(frame: pd.DataFrame, config: ObjectiveConfig | None = None) -> pd.DataFrame:
    out = frame.copy()
    out["expectancy_r"] = out["avg_r"]
    out["is_score"] = composite_score(out, config)
    for name in ALTERNATIVE_OBJECTIVES:
        column = "is_score" if name == "composite" else name
        # ties broken by the t-statistic of mean R, then by configuration order (deterministic)
        order = out.sort_values([column, "t_stat_r"], ascending=[False, False], kind="stable").index
        ranks = pd.Series(range(1, len(out) + 1), index=order)
        out[f"rank_{name}"] = ranks.reindex(out.index).astype(int)
    return out
