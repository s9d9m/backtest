"""Smoke test the Streamlit dashboard headlessly: load synthetic data, run a backtest and a small optimization."""

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
