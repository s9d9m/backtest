"""Shared helpers: build synthetic sessions whose backtest outcome is known exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orb_lab.engine.data_quality import check_and_clean
from orb_lab.engine.instruments import Instrument
from orb_lab.engine.session_data import prepare_data
from orb_lab.engine.sessions import TradingCalendar, hhmm_to_minutes

ET = "America/New_York"


def make_instrument(**overrides) -> Instrument:
    spec = dict(
        symbol="TEST",
        name="Test future",
        tick_size=0.25,
        tick_value=12.5,
        multiplier=50,
        commission_per_side=0.0,
        exchange_fee_per_side=0.0,
        slippage_ticks=0.0,
        calendar="WEEKDAYS",
    )
    spec.update(overrides)
    return Instrument(**spec)


def day_bars(date: str, bars: dict[str, tuple], start: str = "09:00", end: str = "16:00", first_price: float = 100.0) -> pd.DataFrame:
    """1-minute bars for one session from ``start`` to ``end`` ET (bar-start labels).

    ``bars`` maps "HH:MM" -> (open, high, low, close). Unspecified minutes are flat bars at the previous
    close, so a path is described by just its interesting bars.
    """
    s, e = hhmm_to_minutes(start), hhmm_to_minutes(end)
    rows = []
    last = first_price
    for m in range(s, e):
        key = f"{m // 60:02d}:{m % 60:02d}"
        if key in bars:
            o, h, l, c = bars[key]
        else:
            o = h = l = c = last
        rows.append((pd.Timestamp(date) + pd.Timedelta(minutes=m), o, h, l, c, 100))
        last = c
    frame = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")
    frame.index = frame.index.tz_localize(ET).tz_convert("UTC")
    return frame


def prepare(frames, instrument: Instrument | None = None, calendar: str = "WEEKDAYS", **kwargs):
    instrument = instrument or make_instrument()
    bars = pd.concat(frames if isinstance(frames, list) else [frames]).sort_index()
    clean, report = check_and_clean(bars, instrument, bar_minutes=1, calendar=TradingCalendar(calendar))
    assert report.ok, report.to_frame()
    return prepare_data(clean, instrument, calendar=TradingCalendar(calendar), **kwargs)


# A standard range: 09:30-09:45 ET, OR high 101, OR low 99 (width 2.0 points = 8 ticks)
RANGE_BARS = {"09:30": (100.0, 101.0, 99.0, 100.0)}


@pytest.fixture
def instrument():
    return make_instrument()


@pytest.fixture(autouse=True)
def _isolated_research_dir(tmp_path, monkeypatch):
    """Tests must never touch the real research registry / lockbox ledger."""
    monkeypatch.setenv("ORB_RESEARCH_DIR", str(tmp_path / "research"))
