"""Plain-language explanations of every headline metric, with sample-size cautions."""

from __future__ import annotations

import math

import pandas as pd

# key -> (label, format, explanation)
GLOSSARY: dict[str, tuple[str, str, str]] = {
    "n_trades": ("Trades", "int", "Number of executed trades. Under ~30 trades almost every statistic is dominated by luck; "
                 "for a serious conclusion you want hundreds."),
    "win_rate": ("Win rate", "pct", "Share of trades with positive net P&L. Meaningless on its own: a 30 % win rate can be "
                 "profitable with large winners, a 70 % win rate can lose money with large losers."),
    "avg_r": ("Expectancy (R)", "r", "Average net result per trade in units of the initial risk (R = entry-to-stop distance). "
              "+0.10 R means you make on average 10 % of what you risk per trade, after costs."),
    "expectancy": ("Expectancy ($)", "usd", "Average net $ per trade at the chosen position size."),
    "profit_factor": ("Profit factor", "x", "Gross profits / gross losses (after costs). Above 1 = net profitable; "
                      "values above ~1.3 over hundreds of trades are notable; infinite = no losing trades (tiny sample)."),
    "avg_winner": ("Average winner ($)", "usd", "Mean net P&L of winning trades."),
    "avg_loser": ("Average loser ($)", "usd", "Mean net P&L of losing trades (negative)."),
    "payoff_ratio": ("Payoff ratio", "x", "Average winner / |average loser|. Together with win rate it determines expectancy."),
    "net_pnl": ("Net P&L ($)", "usd", "Total profit after commissions, fees, friction and slippage."),
    "total_return": ("Net return", "pct", "Net P&L / starting equity."),
    "cagr": ("CAGR", "pct", "Compound annual growth rate. Annualising a short test multiplies noise: ignore it below one year."),
    "sharpe": ("Sharpe ratio (annualised)", "x2", "Mean daily return / daily volatility x sqrt(252), over every session "
               "(flat days included). Unreliable below one year of data; for context, a Sharpe above 1 sustained out of sample "
               "is rare."),
    "sortino": ("Sortino ratio (annualised)", "x2", "Like Sharpe but penalises only downside volatility. Same sample-size caveat."),
    "max_dd": ("Max drawdown", "pct", "Largest fall from a previous equity peak. The worst historical drawdown is usually NOT the "
               "worst you will experience (see Monte Carlo)."),
    "calmar": ("Calmar ratio", "x2", "CAGR / |max drawdown|. Inherits CAGR's short-sample problem."),
    "t_stat_r": ("t-statistic of mean R", "x2", "Mean R / standard error. Roughly: |t| < 2 is not distinguishable from zero; "
                 "and after searching many configurations even t = 2-3 is expected by chance (multiple testing)."),
    "top5_share": ("Top-5 trade share", "pct", "Share of total profit contributed by the 5 best trades. Above ~50 % means the "
                   "result depends on a handful of outliers."),
    "pnl_ex_top5": ("Net P&L excluding top 5 trades ($)", "usd", "What remains if the 5 best trades had not happened. "
                    "Negative = the edge is carried by outliers."),
    "n_long": ("Long trades", "int", "Number of long trades."),
    "n_short": ("Short trades", "int", "Number of short trades."),
    "pnl_long": ("Long net P&L ($)", "usd", "Net P&L of long trades. A result that only comes from one side may reflect the "
                 "market's trend during the sample rather than the breakout."),
    "pnl_short": ("Short net P&L ($)", "usd", "Net P&L of short trades."),
    "expectancy_r_long": ("Long expectancy (R)", "r", "Average R of long trades."),
    "expectancy_r_short": ("Short expectancy (R)", "r", "Average R of short trades."),
    "cost_per_trade": ("Cost per trade ($)", "usd", "Commission + exchange/regulatory fees + modelled friction + slippage, per trade "
                       "at the chosen size."),
    "total_cost": ("Total costs ($)", "usd", "All transaction costs paid over the test."),
    "ambiguous_exit_pct": ("Ambiguous exits", "pct", "Trades whose exit bar contained both stop and target, so the order of the two "
                           "is unknown from OHLC bars. Resolved with the chosen ambiguity assumption; a high share makes results "
                           "depend on that assumption."),
    "longest_loss_streak": ("Longest losing streak", "int", "Most consecutive losing trades."),
    "cum_r": ("Cumulative R", "r", "Sum of all trade R multiples."),
}


def fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "usd":
        return f"${value:,.2f}"
    if kind == "r":
        return f"{value:+.3f} R"
    if kind == "x":
        return f"{value:.2f}"
    return f"{value:.2f}"


def cautions(metrics: dict) -> list[str]:
    """Sample-size and interpretation warnings for a metrics dict (from ``compute_metrics``)."""
    out = []
    n = int(metrics.get("n_trades", 0) or 0)
    years = float(metrics.get("years", 0) or 0)
    if n == 0:
        return ["No trades: nothing can be concluded."]
    if n < 30:
        out.append(f"Only {n} trades: win rate, expectancy and profit factor can swing wildly with a few trades. Not evidence.")
    elif n < 100:
        out.append(f"{n} trades is a small sample; treat every statistic as very uncertain.")
    if years < 1:
        out.append(f"The test covers {years * 12:.1f} months. Annualised figures (CAGR, Sharpe, Sortino, Calmar) extrapolate a short "
                   "sample and are unreliable; prefer expectancy (R), t-statistic and trade counts.")
    t = abs(float(metrics.get("t_stat_r", 0) or 0))
    if t < 2:
        out.append(f"|t| = {t:.2f} < 2: the average trade is not statistically distinguishable from zero.")
    top5 = metrics.get("top5_share")
    if top5 is not None and not (isinstance(top5, float) and math.isnan(top5)) and top5 > 0.5:
        out.append(f"The 5 best trades produced {top5:.0%} of the profit: results depend on outliers.")
    nl, ns = metrics.get("n_long", 0), metrics.get("n_short", 0)
    pl, ps = metrics.get("pnl_long", 0.0), metrics.get("pnl_short", 0.0)
    if nl and ns and (pl > 0) != (ps > 0):
        out.append("Only one side (long or short) is profitable; this may reflect the sample's market direction.")
    if float(metrics.get("ambiguous_exit_pct", 0) or 0) > 0.1:
        out.append("More than 10 % of exits are ambiguous (stop and target in the same bar): results depend on the ambiguity assumption.")
    return out


ANNUALISED = {"cagr", "sharpe", "sortino", "calmar"}


def metric_table(metrics: dict, keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys or list(GLOSSARY)
    short = float(metrics.get("years", 1) or 0) < 1
    rows = []
    for k in keys:
        if k not in metrics or k not in GLOSSARY:
            continue
        label, kind, text = GLOSSARY[k]
        note = "annualised from < 1 year: unreliable" if (short and k in ANNUALISED) else ""
        rows.append({"metric": label, "value": fmt(metrics[k], kind), "what it means": text, "caution": note})
    return pd.DataFrame(rows)


HEADLINE = ["n_trades", "win_rate", "avg_r", "expectancy", "profit_factor", "avg_winner", "avg_loser", "payoff_ratio", "net_pnl",
            "total_return", "cagr", "sharpe", "sortino", "max_dd", "calmar", "t_stat_r", "top5_share", "pnl_ex_top5", "n_long",
            "n_short", "pnl_long", "pnl_short", "expectancy_r_long", "expectancy_r_short", "cost_per_trade", "total_cost",
            "ambiguous_exit_pct", "longest_loss_streak", "cum_r"]
