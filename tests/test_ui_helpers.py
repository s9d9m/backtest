"""Pure helpers behind the dashboard: number formatting, robust-candidate choice, fold consistency, degradation,
Monte Carlo headline numbers and trade filters."""

import numpy as np
import pandas as pd
import pytest

from orb_lab.engine.params import StrategyParams
from orb_lab.optimization.monte_carlo import MCConfig, run_monte_carlo
from orb_lab.ui import logic as L
from orb_lab.ui import theme as T


def test_number_formats_match_the_house_style():
    assert T.usd(43532.4) == "$43,532"
    assert T.usd(-1294.6) == "−$1,295"
    assert T.usd(12.0, signed=True) == "+$12"
    assert T.usd(-0.2) == "$0"  # no "negative zero"
    assert T.pct(0.088) == "8.8%" and T.pct(-0.093) == "−9.3%" and T.pct(0.05, signed=True) == "+5.0%"
    assert T.rmult(0.1312) == "+0.13R" and T.rmult(-0.0249) == "−0.02R" and T.rmult(0.001) == "0.00R"
    assert T.ratio(1.6412) == "1.64" and T.ratio(float("inf")) == "∞"
    for f in (T.usd, T.pct, T.rmult, T.ratio, T.count):
        assert f(None) == "—" and f(float("nan")) == "—"


def test_glossary_covers_the_terms_a_beginner_meets():
    for term in ("Expectancy", "Profit factor", "Sharpe", "Sortino", "Max drawdown", "Calmar", "t-stat", "Top-5 share", "OOS", "WFO",
                 "Monte Carlo"):
        assert T.TIPS.get(term), term
    assert 'title="' in T.tip("Sharpe")


def _grid():
    rows = []
    for i, (pnl, r, n, stab, rank) in enumerate([(900.0, 0.40, 12, "spike", 3), (500.0, 0.20, 60, "plateau", 2), (450.0, 0.25, 55, "mixed", 1),
                                                 (300.0, 0.10, 80, "plateau", 4), (-100.0, -0.05, 90, "plateau", 5)]):
        rows.append({"config_key": f"k{i}", "net_pnl": pnl, "avg_r": r, "n_trades": n, "stability": stab, "rank_composite": rank})
    return pd.DataFrame(rows)


def test_best_historical_and_robust_candidate_differ_when_the_best_is_a_spike():
    g = _grid()
    assert L.best_historical(g)["config_key"] == "k0"  # highest P&L, but 12 trades on a spike
    row, why = L.robust_candidate(g, min_trades=20)
    assert row["config_key"] == "k1" and "neighbouring" in why  # best-ranked plateau with enough trades
    row, _ = L.robust_candidate(g[g["stability"] != "plateau"], min_trades=20)
    assert row["config_key"] == "k2"  # falls back to 'mixed'
    row, why = L.robust_candidate(g[g["stability"] == "spike"], min_trades=20)
    assert row is None and "no configuration" in why
    assert L.robust_candidate(pd.DataFrame())[0] is None


def test_fold_consistency_and_degradation():
    w = pd.DataFrame({"window": [0, 1, 2, 3], "selected": [3, -1, 5, 7], "oos_exp_r": [0.1, np.nan, -0.2, 0.05]})
    assert L.fold_consistency(w) == {"folds": 3, "positive": 2, "share": pytest.approx(2 / 3)}
    d = L.degradation({"median_train_exp_r": 0.2, "median_val_exp_r": 0.1, "median_oos_exp_r": 0.05})
    assert d["drop_share"] == pytest.approx(0.75)
    assert np.isnan(L.degradation({"median_train_exp_r": -0.1, "median_oos_exp_r": 0.05})["drop_share"])


def test_monte_carlo_headline_numbers():
    r = np.array([1.0, -1.0, 0.5, -1.0, 2.0, -1.0, 1.0, -0.5, 1.5, -1.0] * 3)
    res = run_monte_carlo(r, MCConfig(n_sims=2000, seed=7, starting_equity=40_000, risk_pct=0.02, sizing="compounding"))
    h = L.mc_headline(res)
    assert h["start"] == 40_000 and h["hist_end"] == pytest.approx(res.historical["ending_equity"])
    assert h["p5_end"] <= h["median_end"] <= h["p95_end"]
    assert h["p5_dd"] <= h["median_dd"] <= 0
    assert 0 <= h["p_loss"] <= 1 and h["median_streak"] <= h["p95_streak"]


def test_trade_filters():
    t = pd.DataFrame({"session_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
                      "direction": ["long", "short", "long", "short"], "net_pnl": [100.0, -50.0, 0.0, 30.0],
                      "r_multiple": [1.0, -0.5, 0.0, 0.3], "entry_method": ["market", "market", "stop", "market"],
                      "market": ["SPY"] * 4, "sized_out": [False, False, False, True]})
    assert len(L.filter_trades(t)) == 3  # skipped (sized-out) signals are never shown as trades
    assert L.filter_trades(t, sides=["long"])["direction"].eq("long").all()
    assert L.filter_trades(t, outcome="winners")["net_pnl"].gt(0).all()
    assert L.filter_trades(t, outcome="losers")["net_pnl"].le(0).all()
    assert L.filter_trades(t, methods=["stop"])["entry_method"].eq("stop").all()
    assert len(L.filter_trades(t, start="2024-01-03", end="2024-01-04")) == 2
    assert len(L.filter_trades(t, r_range=(0.5, 2.0))) == 1


def test_strategy_description_is_plain_language():
    from orb_lab.ui import state as S

    p = StrategyParams(range_minutes=15, entry_tf=15, entry_method="market", stop_method="or_mid", target_r=0.75)
    text = S.strategy_one_liner(p, "QQQ")
    assert text.startswith("QQQ · 09:30 ORB · 15m range · 15m entry · Market · Midpoint stop · 0.75R target")
    assert S.params_text(p).startswith("09:30 ORB")
    assert S.stop_text(StrategyParams(stop_method="or_pct", stop_param=0.5)) == "50% of range stop"


def test_prices_show_every_tick():
    assert T.price_decimals(0.00005) == 5 and T.price(1.15405, 0.00005) == "1.15405"  # 6E
    assert T.price_decimals(0.25) == 2 and T.price(5123.25, 0.25) == "5,123.25"  # ES
    assert T.price_decimals(0.10) == 2 and T.price_decimals(0.01) == 2
