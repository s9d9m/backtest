"""Smoke test the Streamlit dashboard headlessly: load synthetic data, run a backtest and a small optimization."""

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _button(at, label):
    for b in at.button:
        if b.label == label:
            return b
    for b in at.sidebar.button:
        if b.label == label:
            return b
    raise AssertionError(f"button {label!r} not found")


def test_dashboard_end_to_end():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Synthetic").run()
    at.sidebar.text_input[0].set_value("2021-01-04")
    at.sidebar.text_input[1].set_value("2021-12-30")
    _button(at, "Load data").click().run()
    assert not at.exception, at.exception
    assert any("tradable sessions" in s.value for s in at.sidebar.success)
    _button(at, "RUN BACKTEST").click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Net P&L" for m in at.metric)
    _button(at, "Run execution stress test").click().run()
    assert not at.exception, at.exception
    assert any(("ROBUST" in (x.value or "")) or ("FRAGILE" in (x.value or "")) or ("NOT POSITIVE" in (x.value or ""))
               for x in list(at.success) + list(at.error))
    # reduce the preset to a tiny grid before running the optimization
    for ms in at.multiselect:
        if ms.label == "Target R":
            ms.set_value([1.0])
        if ms.label == "Range length":
            ms.set_value([15, 30])
        if ms.label == "Entry TF":
            ms.set_value([5])
        if ms.label == "Stop":
            ms.set_value(["or_mid", "or_opposite"])
        if ms.label == "Cutoff":
            ms.set_value(["11:00"])
    for ni in at.number_input:
        if ni.label == "Worker processes":
            ni.set_value(1)
    at.run()
    _button(at, "RUN OPTIMIZATION").click().run()
    assert not at.exception, at.exception
    assert any("configurations have positive Sharpe" in (w.value or "") for w in list(at.warning) + list(at.info))
    assert any(m.label == "Effective independent trials" for m in at.metric)


def test_phase0_tab_and_free_yahoo_source_offline(tmp_path):
    """The default view shows committed Phase-0 results; 'Free Yahoo' loads a saved copy with the ETF spec (no network)."""
    from orb_lab.data_sources import yahoo

    from .test_phase0 import FakeTicker, yahoo_frame

    dates = [str(d.date()) for d in pd.bdate_range("2024-06-03", "2024-06-28") if str(d.date()) != "2024-06-19"]
    yahoo.download("SPY", "5m", dates[0], "2024-06-29", chunk_days=40, ticker=FakeTicker(yahoo_frame(dates)))  # into the isolated DATA_DIR
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Unseen-test trades" in labels and "Conclusion" in labels
    assert at.sidebar.radio[0].value == "Free Yahoo (SPY/QQQ)"
    _button(at, "Load data").click().run()
    assert not at.exception, at.exception
    assert any("tradable sessions" in s.value for s in at.sidebar.success)
    assert any("NOT FUTURES VALIDATION" in w.value for w in at.warning)
    _button(at, "RUN BACKTEST").click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Net P&L" for m in at.metric)
    # ETF spec, not a futures spec: $0.02 per SHARE per side friction, no commission, and 100 shares by default
    assert any(ni.label == "Modelled bid/ask friction ($ per share per side)" and abs(ni.value - 0.02) < 1e-9 for ni in at.number_input)
    assert any(ni.label == "Commission ($ per share per side)" and ni.value == 0 for ni in at.number_input)
    assert not any("per contract" in (ni.label or "") for ni in at.number_input)
    assert any(ni.label == "Fixed quantity (shares)" and ni.value == 100 for ni in at.number_input)
    tfs = next(sb for sb in at.selectbox if sb.label == "Entry timeframe (min)")
    assert set(tfs.options) == {"5", "10", "15"}


def _set(widgets, label, value):
    for w in widgets:
        if w.label == label:
            w.set_value(value)
            return w
    raise AssertionError(f"widget {label!r} not found")


def _texts(at):
    return [x.value or "" for x in list(at.success) + list(at.info) + list(at.warning) + list(at.error)]


def test_full_research_pipeline_in_the_dashboard(tmp_path):
    """Synthetic data through every stage in the browser: backtest -> optimize -> candidate -> robustness -> Monte Carlo
    -> split/validate/freeze -> walk-forward job -> blind holdout -> report."""
    import json
    import time

    from orb_lab import jobs

    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    at.sidebar.radio[0].set_value("Synthetic").run()
    at.sidebar.text_input[0].set_value("2021-01-04")
    at.sidebar.text_input[1].set_value("2021-12-30")
    _button(at, "Load data").click().run()
    assert not at.exception, at.exception
    assert any("SYNTHETIC DATA" in t for t in _texts(at))
    # reserve the blind holdout BEFORE optimisation
    _button(at, "Save split and withhold the holdout").click().run()
    assert not at.exception, at.exception
    assert any("Blind holdout" in t and "withheld" in t for t in _texts(at))
    # risk-based sizing: 1 % of $40,000
    _set(at.selectbox, "Position sizing", "pct_equity")
    _button(at, "RUN BACKTEST").click().run()
    assert not at.exception, at.exception
    # tiny optimization on the train period
    for label, value in (("Target R", [1.0, 1.5]), ("Range length", [15, 30]), ("Entry TF", [5]), ("Stop", ["or_mid", "or_opposite"]),
                         ("Cutoff", ["11:00"])):
        _set(at.multiselect, label, value)
    for ni in at.number_input:
        if ni.label == "Worker processes":
            ni.set_value(1)
    at.run()
    _button(at, "RUN OPTIMIZATION").click().run()
    assert not at.exception, at.exception
    for b in at.button:
        if b.key == "opt_to_cand":
            b.click()
    at.run()
    assert not at.exception, at.exception
    assert any("Saved as candidate" in t for t in _texts(at))
    # robustness: only two axes to keep the test fast
    _set(at.multiselect, "Parameters to vary (one at a time; everything else fixed at the candidate)", ["target_r", "range_minutes"])
    _button(at, "RUN NEIGHBOURHOOD SWEEP").click().run()
    assert not at.exception, at.exception
    assert any(t.startswith("Verdict:") for t in _texts(at))
    # Monte Carlo on the candidate
    _set(at.number_input, "Number of simulations", 500)
    _button(at, "RUN MONTE CARLO").click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Probability of ending with a loss" for m in at.metric)
    # validate + freeze + blind test
    _button(at, "Evaluate on train and validation").click().run()
    assert not at.exception, at.exception
    for b in at.button:
        if b.key == "ps_val_use":
            b.click()
    at.run()
    _button(at, "FREEZE (writes an immutable file with a SHA-256 hash)").click().run()
    assert not at.exception, at.exception
    assert any("Frozen" in t and "SHA-256" in t for t in _texts(at))
    _set(at.checkbox, "I will not change the candidate after seeing the result.", True)
    at.run()
    _button(at, "RUN BLIND HOLDOUT TEST").click().run()
    assert not at.exception, at.exception
    assert any(t.startswith("BLIND test of") for t in _texts(at))
    # walk-forward launched from the browser (session windows on this short sample), run to completion
    _set(at.radio, "Window unit", "Trading sessions (short samples, e.g. the free ~60-day data)")
    _set(at.number_input, "Train sessions", 60)
    _set(at.number_input, "Validation sessions", 20)
    _set(at.number_input, "OOS = roll sessions", 20)
    _set(at.selectbox, "Parameter space", "smoke")
    _set(at.number_input, "Min trades in train", 5)
    _set(at.number_input, "Min trades in validation", 2)
    at.run()
    _button(at, "LAUNCH WALK-FORWARD (runs in the background)").click().run()
    assert not at.exception, at.exception
    job = jobs.list_jobs(kind="wfo")[0]
    for _ in range(300):
        if jobs.read_status(job)["state"] in jobs.TERMINAL:
            break
        time.sleep(1)
    status = jobs.read_status(job)
    assert status["state"] == "done", status
    spec = json.loads((job / "spec.json").read_text())
    assert spec["structures"][0]["unit"] == "sessions" and spec["data"]["synthetic"]
    windows = [p for p in (job / "result").glob("*/*/windows.csv")]
    assert windows
    w = pd.read_csv(windows[0])
    # every OOS window starts after its validation window ends, and parameters were frozen (hashed) before OOS
    assert (pd.to_datetime(w["oos_start"]) > pd.to_datetime(w["val_end"])).all()
    assert w.loc[w["selected"] >= 0, "frozen_sha256"].str.len().eq(64).all()
    # the walk-forward OOS never reaches into the blind holdout
    split = at.session_state["split"]
    assert pd.to_datetime(w["oos_end"]).max() < pd.Timestamp(split["holdout_start"])
    at.run()
    assert not at.exception, at.exception
    assert any(t.startswith("Verdict: **") for t in _texts(at))
    assert any("Download report" in (b.label or "") for b in at.get("download_button"))
