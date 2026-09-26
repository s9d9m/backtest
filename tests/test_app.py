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
    _button(at, "Run slippage sensitivity (0-3 ticks)").click().run()
    assert not at.exception, at.exception
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
    assert any("configurations tested" in (w.value or "") for w in list(at.warning) + list(at.success))


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
    # ETF spec, not a futures spec: $0.02/share cost and 100 shares by default
    assert any(ni.label == "Commission/side ($)" and abs(ni.value - 0.02) < 1e-9 for ni in at.number_input)
    assert any(ni.label == "Shares (fixed)" and ni.value == 100 for ni in at.number_input)
    tfs = next(sb for sb in at.selectbox if sb.label == "Entry timeframe (min)")
    assert set(tfs.options) == {"5", "10", "15"}
