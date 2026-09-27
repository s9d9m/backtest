"""Headless dashboard tests (Streamlit AppTest) for the page-based UI.

Navigation is research-first: Overview · Strategy · Backtest · Optimize · Validate · Stress test · Results ·
Trade explorer · Data · Settings & research.
"""

import json
import time
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["overview", "strategy", "backtest", "optimize", "validate", "stress", "results", "trades", "data", "settings"]


def _button(at, label=None, key=None):
    for b in list(at.button) + list(at.sidebar.button):
        if (label is not None and b.label == label) or (key is not None and b.key == key):
            return b
    raise AssertionError(f"button {label or key!r} not found")


def _widget(widgets, label):
    for w in widgets:
        if w.label == label:
            return w
    raise AssertionError(f"widget {label!r} not found")


def _texts(at):
    return [x.value or "" for x in list(at.success) + list(at.info) + list(at.warning) + list(at.error)]


def _html(at):
    return " ".join(m.value for m in at.markdown)


def _go(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception


def _load_synthetic(at, start="2021-01-04", end="2021-12-30"):
    _go(at, "data")
    _widget(at.radio, "Source").set_value("Synthetic (testing only)").run()
    _widget(at.text_input, "Start").set_value(start)
    _widget(at.text_input, "End").set_value(end)
    at.run()
    _button(at, "Load data").click().run()
    assert not at.exception, at.exception
    assert any("tradable sessions" in t for t in _texts(at))


def test_navigation_is_research_first_and_every_page_renders():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception
    nav = at.sidebar.radio[0]
    assert list(nav.options)[0].endswith("Overview") and list(nav.options)[-2].endswith("Data")
    assert nav.value == "overview"
    _load_synthetic(at)
    for page in PAGES:
        _go(at, page)
    _go(at, "overview")
    html = _html(at)
    for stage in ("In-sample", "Validation", "Blind OOS", "Walk-forward", "Robustness", "Execution", "Monte Carlo"):
        assert stage in html
    assert "INSUFFICIENT EVIDENCE" in html  # a backtest alone never validates a strategy
    assert "Ending balance" in html and "Expectancy" in html
    assert any("SYNTHETIC DATA" in t for t in _texts(at))


def test_no_data_shows_a_welcome_instead_of_errors():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    assert "Welcome" in _html(at)
    assert any(b.label == "Load free 6E data (no account)" for b in at.button)
    for page in PAGES:
        _go(at, page)


def test_free_yahoo_data_autoloads_with_the_etf_spec(tmp_path):
    """A saved free SPY copy is loaded on first open; costs are per SHARE and entry timeframes follow the 5-minute bars."""
    from orb_lab.data_sources import yahoo

    from .test_phase0 import FakeTicker, yahoo_frame

    dates = [str(d.date()) for d in pd.bdate_range("2024-06-03", "2024-06-28") if str(d.date()) != "2024-06-19"]
    yahoo.download("SPY", "5m", dates[0], "2024-06-29", chunk_days=40, ticker=FakeTicker(yahoo_frame(dates)))  # isolated DATA_DIR
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    assert any("NOT FUTURES VALIDATION" in t for t in _texts(at))
    assert "Ending balance" in _html(at)
    _go(at, "strategy")
    assert _widget(at.number_input, "Bid/ask friction ($ per share per side)").value == pytest.approx(0.02)
    assert _widget(at.number_input, "Commission ($ per share per side)").value == 0
    assert not any("per contract" in (ni.label or "") for ni in at.number_input)
    assert _widget(at.number_input, "Fixed quantity (shares)").value >= 1
    tfs = _widget(at.selectbox, "Entry timeframe (signal bars)")
    assert set(tfs.options) == {"5-minute bars", "10-minute bars", "15-minute bars"}
    _go(at, "backtest")
    assert "Total P&amp;L" in _html(at)
    _go(at, "data")
    html = _html(at)
    assert "SPY" in html and "not downloaded" in html and "OHLCV-1m" in html


def test_strategy_editor_updates_the_current_strategy():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    _load_synthetic(at, end="2021-06-30")
    _go(at, "strategy")
    _widget(at.selectbox, "Range length").set_value(30)
    _widget(at.selectbox, "Profit target").set_value(1.5)
    _widget(at.selectbox, "Risk per trade").set_value("2%")
    _button(at, "Save strategy").click().run()
    assert not at.exception, at.exception
    strat = at.session_state["strategy"]
    assert strat["params"].range_minutes == 30 and strat["params"].target_r == 1.5
    assert strat["sizing"].risk_pct == pytest.approx(0.02) and strat["sizing"].starting_equity == 40_000
    html = _html(at)
    assert "30m range" in html and "1.5R target" in html and "2% risk/trade" in html


def test_full_research_workflow_in_the_dashboard():
    """Synthetic data through every stage: holdout → backtest → optimize → use candidate → robustness → execution stress →
    Monte Carlo → validation → freeze → blind test → walk-forward job → results → trade explorer."""
    from orb_lab import jobs

    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    _load_synthetic(at)
    # one full ES contract often risks more than 1 % of $40,000: trade a fixed 1 contract so no signal is skipped
    _go(at, "strategy")
    _widget(at.selectbox, "Method").set_value("fixed_contracts")
    _button(at, "Save strategy").click().run()
    assert at.session_state["strategy"]["sizing"].mode == "fixed_contracts"
    _go(at, "validate")
    _button(at, "Save split and withhold the holdout").click().run()
    assert not at.exception, at.exception
    split = at.session_state["split"]
    assert split["n_holdout"] > 0
    # the development pages never see the holdout
    assert str(at.session_state["dataset"].prep.dates[-1]) < split["holdout_start"]
    _go(at, "backtest")
    assert "Ending balance" in _html(at)
    _go(at, "optimize")
    for label, value in (("Target R", [1.0, 1.5]), ("Range length", [15, 30]), ("Entry TF", [5]), ("Stop", ["or_mid", "or_opposite"]),
                         ("Cutoff", ["11:00"])):
        _widget(at.multiselect, label).set_value(value)
    _widget(at.number_input, "Worker processes").set_value(1)
    at.run()
    _button(at, "RUN OPTIMIZATION").click().run()
    assert not at.exception, at.exception
    html = _html(at)
    assert "Best historical configuration" in html and "Robust candidate" in html
    assert any("Multiple testing" in t for t in _texts(at))
    assert at.session_state["grid_period"][1] == split["train_end"]  # optimiser defaulted to the train period
    _button(at, key="pick_use").click().run()
    assert not at.exception, at.exception
    assert at.session_state["strategy"]["source"].startswith("optimizer rank")
    _go(at, "stress")
    _widget(at.multiselect, "Parameters").set_value(["target_r", "range_minutes"])
    at.run()
    _button(at, "RUN NEIGHBOURHOOD SWEEP").click().run()
    assert not at.exception, at.exception
    assert any(k in _html(at) for k in ("BROAD PLATEAU", "MODERATE", "SPIKE / FRAGILE", "NOT POSITIVE"))
    _button(at, "RUN EXECUTION STRESS TEST").click().run()
    assert not at.exception, at.exception
    assert any(k in _html(at) for k in ("ROBUST", "FRAGILE", "NOT POSITIVE"))
    _widget(at.select_slider, "Simulations").set_value(1000)
    at.run()
    _button(at, "RUN MONTE CARLO").click().run()
    assert not at.exception, at.exception
    html = _html(at)
    for card in ("Historical ending balance", "Median simulated ending balance", "Probability of loss", "Bad case ending balance",
                 "Historical max drawdown", "Median simulated max drawdown", "Bad-case drawdown"):
        assert card in html, card
    _go(at, "validate")
    _button(at, "Run validation check").click().run()
    assert "Validation expectancy" in _html(at)
    _button(at, "FREEZE (writes an immutable file with a SHA-256 hash)").click().run()
    assert not at.exception, at.exception
    assert "The current strategy is frozen" in _html(at)
    _widget(at.checkbox, "I will not change the strategy after seeing the result.").check().run()
    _button(at, "RUN BLIND HOLDOUT TEST").click().run()
    assert not at.exception, at.exception
    assert any(t.startswith("**BLIND** test of") for t in _texts(at))
    # walk-forward in session windows, launched from the browser and run to completion
    _widget(at.radio, "Window unit").set_value("Trading sessions (short samples, e.g. the free ~60-day data)")
    _widget(at.number_input, "Train sessions").set_value(60)
    _widget(at.number_input, "Validation sessions").set_value(20)
    _widget(at.number_input, "OOS = roll sessions").set_value(20)
    _widget(at.selectbox, "Parameter space searched in each fold").set_value("smoke")
    _widget(at.number_input, "Min trades in train").set_value(5)
    _widget(at.number_input, "Min trades in validation").set_value(2)
    at.run()
    _button(at, "LAUNCH WALK-FORWARD (runs in the background)").click().run()
    assert not at.exception, at.exception
    job = jobs.list_jobs(kind="wfo")[0]
    for _ in range(300):
        if jobs.read_status(job)["state"] in jobs.TERMINAL:
            break
        time.sleep(1)
    assert jobs.read_status(job)["state"] == "done", jobs.read_status(job)
    spec = json.loads((job / "spec.json").read_text())
    assert spec["structures"][0]["unit"] == "sessions" and spec["data"]["synthetic"]
    w = pd.read_csv(next((job / "result").glob("*/*/windows.csv")))
    assert (pd.to_datetime(w["oos_start"]) > pd.to_datetime(w["val_end"])).all()
    assert w.loc[w["selected"] >= 0, "frozen_sha256"].str.len().eq(64).all()
    assert pd.to_datetime(w["oos_end"]).max() < pd.Timestamp(split["holdout_start"])  # WFO never touches the holdout
    at.run()
    assert "Stitched blind-OOS performance" in _html(at)
    _go(at, "results")
    html = _html(at)
    assert any(v in html for v in ("PROMISING", "MIXED", "NO PRELIMINARY EVIDENCE", "INSUFFICIENT EVIDENCE"))
    assert "Blind holdout" in html and "Walk-forward OOS" in html
    assert any("Download report" in (b.label or "") for b in at.get("download_button"))
    _go(at, "trades")
    at.segmented_control[0].set_value("blind").run()
    assert not at.exception, at.exception
    assert "Trades shown" in _html(at)
    assert any("Download filtered trades" in (b.label or "") for b in at.get("download_button"))
    _go(at, "overview")
    assert "Blind OOS" in _html(at)
    _go(at, "settings")
    assert not at.exception


def test_free_6e_futures_data_uses_the_futures_spec(tmp_path):
    """Free Yahoo 6E=F bars load as Euro FX futures: contracts, $6.25 per 0.00005 tick, futures costs; 6E is preferred on autoload."""
    from orb_lab.data_sources import yahoo

    from .test_phase0 import FakeTicker, yahoo_frame

    dates = [str(d.date()) for d in pd.bdate_range("2024-06-03", "2024-06-28") if str(d.date()) != "2024-06-19"]
    frame = yahoo_frame(dates) / 400.0  # EUR/USD-like price level
    frame["Volume"] = 1000
    for sym, f in (("SPY", yahoo_frame(dates)), ("6E", frame)):
        yahoo.download(sym, "5m", dates[0], "2024-06-29", chunk_days=40, ticker=FakeTicker(f), data_dir=yahoo.DATA_DIR / "dashboard_yahoo",
                       decimals=6 if sym == "6E" else 4)
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    inst = at.session_state["instrument"]
    assert inst.symbol == "6E" and inst.unit == "contract" and inst.tick_size == 0.00005 and inst.tick_value == 6.25
    assert any("FREE 6E DATA" in t for t in _texts(at))
    assert not any("FREE PROXY DATA" in t for t in _texts(at))
    _go(at, "strategy")
    assert _widget(at.number_input, "Commission ($ per contract per side)").value == pytest.approx(0.85)
    assert not any("per share" in (ni.label or "") for ni in at.number_input)
    _go(at, "backtest")
    assert "Ending balance" in _html(at)
    _go(at, "data")
    assert "free Yahoo continuous" in _html(at)
